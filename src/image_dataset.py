# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import leveldb
import logging
import os
import random
import threading
import time
from collections import defaultdict
from queue import Queue

import cv2
import mxnet as mx
import numpy as np
from mxnet import io, nd

logger = logging.getLogger()


class FaceDataset(mx.gluon.data.Dataset):
    pic_db_dict = {}

    def __init__(self, leveldb_path, label_path, leveldb_feature_path=None, pic_ignore=os.path.expanduser("~/datasets/cacher/picture.ignore"),
                 min_images=0, max_images=11111111111, ignore_labels=set()):
        super(FaceDataset, self).__init__()
        assert leveldb_path
        logger.info('loading FaceDataset %s %s min_images %s max_images %s', leveldb_path, label_path, min_images, max_images)
        self.leveldb_path = leveldb_path
        self.label_path = label_path
        self.path_imgidx_filter = label_path + ".filter"
        self.path_label_merged = label_path + ".merged"
        self.leveldb_feature_path = leveldb_feature_path

        self.ignore_pic_ids = set()
        if "processed" not in label_path and os.path.exists(pic_ignore):
            pic_ids = open(pic_ignore).readlines()
            pic_ids = [pic_id.strip() for pic_id in pic_ids]
            self.ignore_pic_ids = set(pic_ids)
            logger.info("FaceDataset ignore_pic_ids %s", len(self.ignore_pic_ids))

        self.filter_labels = set()
        if os.path.exists(self.path_imgidx_filter):
            filtered_labels = open(self.path_imgidx_filter).readlines()
            self.filter_labels = [int(l) for l in filtered_labels]
            self.filter_labels = set(self.filter_labels)
            logger.info("FaceDataset filter_labels %s", len(self.filter_labels))

        self.replace_labels = {}
        self.replace_label_set = set()
        if os.path.exists(self.path_label_merged):
            lines = open(self.path_label_merged).readlines()
            for merged_labels in lines:
                labels = [int(l) for l in merged_labels.strip().split(",") if int(l) not in self.filter_labels]
                labels = sorted(labels)
                if len(labels) > 1:
                    assert len(self.replace_label_set.intersection(set(labels))) == 0
                    self.replace_label_set.update(set(labels))
                    for l in labels[:-1]:
                        self.replace_labels[l] = labels[-1]
            logger.info("FaceDataset replace_labels %s", len(self.replace_labels))

        if leveldb_path in self.pic_db_dict:
            self.pic_db = self.pic_db_dict[leveldb_path]
        else:
            self.pic_db = leveldb.LevelDB(leveldb_path, max_open_files=100)
            self.pic_db_dict[leveldb_path] = self.pic_db
        with open(self.label_path, "r") as file:
            lines = file.readlines()
            self.base_pic_ids = []
            self.base_labels = []
            self.base_label2pic = defaultdict(list)
            for index, line in enumerate(lines):
                pic_id, label = line.strip().split(",")
                label = int(label)
                if label in self.replace_labels:
                    label = self.replace_labels[label]
                if label == -1 or label in ignore_labels or label in self.filter_labels or pic_id in self.ignore_pic_ids:
                    continue
                self.base_pic_ids.append(pic_id)
                self.base_labels.append(label)
                self.base_label2pic[label].append(pic_id)
        self.min_images = min_images
        self.max_images = max_images
        self.reset()

    def reset(self):
        logger.info("origin pic_ids %s labels %s", len(self.base_pic_ids), len(self.base_label2pic))
        min_images = self.min_images
        max_images = self.max_images
        if min_images > 0:
            new_label2pic = defaultdict(list)
            for l in self.base_label2pic:
                c = self.base_label2pic[l]
                if len(c) >= min_images:
                    sub = random.sample(c, min(max_images, len(c)))
                    new_label2pic[l] = sub
            new_pic_ids = []
            new_labels = []
            for label in new_label2pic:
                if label in new_label2pic:
                    new_pic_ids += new_label2pic[label]
                    new_labels += [label] * len(new_label2pic[label])
            self.pic_ids = new_pic_ids
            self.labels = new_labels
            self.label2pic = new_label2pic
        else:
            self.pic_ids = self.base_pic_ids
            self.labels = self.base_labels
            self.label2pic = self.base_label2pic

        self.order_labels = sorted(self.label2pic.keys())
        self.train_labels = {}
        for index, label in enumerate(self.order_labels):
            self.train_labels[label] = index
        logger.info("final pic_ids %s labels %s", len(self.pic_ids), len(self.label2pic))

    def is_deleted(self, label):
        return label in self.filter_labels

    def delete_label(self, label):
        self.filter_labels.add(label)
        with open(self.path_imgidx_filter, "w") as file:
            for l in sorted(self.filter_labels, reverse=False):
                file.write(str(l))
                file.write("\n")

    def check_labels(self, random_select, fea_db):
        label2score = {}
        for batch_index, label in enumerate(self.order_labels):
            pic_ids = self.label2pic[label]
            pic_ids = random.sample(pic_ids, min(len(pic_ids), random_select))

            ret_features = nd.empty((len(pic_ids), 512))
            for pic_id_index, pic_id in enumerate(pic_ids):
                try:
                    pic_id = str(pic_id).encode('utf-8')
                    data = fea_db.Get(pic_id)
                    ret_features[pic_id_index][:] = np.frombuffer(data, dtype=np.float32)[:512]
                except Exception as e:
                    logger.info("pic_id %s no features", pic_id)
            scores = nd.dot(ret_features, ret_features.T)
            mean = (nd.sum(scores) - len(scores)) / (len(scores) * len(scores) - len(scores))
            mean = mean.asscalar()
            label2score[label] = mean
            if batch_index % 10 == 0:
                logger.info("mean %s batch_index/count %s/%s", mean, batch_index, self.label_len)
        return label2score

    def mean_pic_ids(self, random_select, fea_db):
        while True:
            label = self.task_queue.get()
            if label is None:
                break
            pic_ids = self.label2pic[label]
            pic_ids = random.sample(pic_ids, min(len(pic_ids), random_select))

            ret_features = nd.empty((len(pic_ids), 512))
            for pic_id_index, pic_id in enumerate(pic_ids):
                try:
                    pic_id = str(pic_id).encode('utf-8')
                    data = fea_db.Get(pic_id)
                    ret_features[pic_id_index][:] = np.frombuffer(data, dtype=np.float32)[:512]
                except Exception as e:
                    logger.info("pic_id %s no features", pic_id)
            scores = nd.dot(ret_features, ret_features.T)
            mean = (nd.sum(scores) - len(scores)) / (len(scores) * len(scores) - len(scores))
            # logger.info("result_queue %s", label)
            self.result_queue.put([label, mean.asscalar()])
        with self.thread_lock:
            self.activated -= 1
            logger.info("thread end %s", self.activated)
        if self.activated <= 0:
            self.result_queue.put(None)

    def check_labels_by_thread(self, random_select, fea_db, thread_count=6):
        self.task_queue = Queue(len(self.order_labels))
        self.result_queue = Queue(len(self.order_labels))
        self.thread_lock = threading.Lock()
        self.activated = thread_count
        for _ in range(thread_count):
            t = threading.Thread(target=self.mean_pic_ids, args=(random_select, fea_db))
            t.start()

        for label in self.order_labels:
            self.task_queue.put(label)
        for _ in range(thread_count):
            self.task_queue.put(None)
        label2score = {}
        process_count = 0
        while True:
            result = self.result_queue.get()
            if result is None:
                break
            process_count += 1
            if process_count % 100 == 0:
                logger.info("process_count %s", process_count)
            label, mean = result
            label2score[label] = mean
        return label2score

    def db_clear(self, pic_ids):
        clear_labels = {}
        for index, pic in enumerate(self.pic_ids):
            if pic in pic_ids:
                label = self.labels[index]
                # clear_labels.append(label)
                clear_labels[label] = 1
                self.delete_label(label)
        return clear_labels

    def save_processed(self):
        path_label_processed = self.label_path + ".processed"
        logger.info("path_label_processed label_len %s order_labels %s", self.label_len, len(self.order_labels))
        assert self.label_len == len(self.order_labels)
        with open(path_label_processed, "w") as file:
            for i in range(self.label_len):
                label = self.order_labels[i]
                pic_ids = self.label2pic[label]
                for pic_id in pic_ids:
                    if label == -1:
                        continue
                    file.write("{},{}\n".format(pic_id, label))

    def check_warning_labels(self, leveldb_feature_path, th=None, random_select=30):
        path_label_check = self.label_path + ".check"
        if not os.path.exists(path_label_check):
            if os.path.exists(leveldb_feature_path):
                fea_db = leveldb.LevelDB(leveldb_feature_path, max_open_files=100)
                # label2score = self.check_labels(random_select, fea_db)
                label2score = self.check_labels_by_thread(random_select, fea_db)

            sections = [0.4, 0.5, 0.6, 0.7, 1]
            counts = [0] * len(sections)
            for k in label2score:
                mean = label2score[k]
                for index, count in enumerate(sections):
                    if mean < count:
                        counts[index] += 1
                        break
            per_counts = [c / len(label2score) for c in counts]
            logger.info("sections %s counts %s per_counts %s", sections, counts, per_counts)

            with open(path_label_check, "w") as file:
                items = list(label2score.items())
                items = sorted(items, key=lambda x: x[1])
                for k, v in items:
                    file.write(str(k) + "," + str(v))
                    file.write("\n")

        label2score = {}
        with open(path_label_check, "r") as file:
            items = file.readlines()
            for line in items:
                label, score_mean = line.strip().split(",")
                label = int(label)
                if th is None:
                    if float(score_mean) < 0.6 and label in self.label2pic:
                        label2score[label] = float(score_mean)
                else:
                    if float(score_mean) < th and label in self.label2pic:
                        self.delete_label(label)
                        label2score[label] = float(score_mean)
        return label2score

    def inner_check(self, th=0.5, percent=0.5, check_count=10, leveldb_feature_path=os.path.expanduser("~/datasets/cacher/features")):
        cache_path = self.label_path + ".inner_check_%.2f" % th
        if not os.path.exists(cache_path):
            if os.path.exists(leveldb_feature_path):
                features = []
                face_ids = []
                face_labels = []
                clusters = defaultdict(list)
                fea_db = leveldb.LevelDB(leveldb_feature_path, max_open_files=100)
                for batch_index, label in enumerate(self.order_labels):
                    if batch_index % 1000 == 0:
                        logger.info("label_features batch_index/count %s/%s", batch_index, self.label_len)
                    pic_ids = self.label2pic[label]
                    pic_ids = random.sample(pic_ids, min(check_count, len(pic_ids)))
                    for index, pic_id in enumerate(pic_ids):
                        clusters[label].append(index)
                        face_ids.append(pic_id)
                        face_labels.append(label)
                        try:
                            pic_id = str(pic_id).encode('utf-8')
                            data = fea_db.Get(pic_id)
                            features.append(np.frombuffer(data, dtype=np.float32)[:512])
                        except Exception as e:
                            logger.info("pic_id %s no pic", pic_id)
                            sys.exit()

                face_features = np.array(features, dtype=np.float32)
                params = cdp.ClusterParams(features=face_features, labels=np.ones(len(face_features)) * -1, th_knn=th, valid_percent=percent)
                clusters_list = list(clusters.values())
                new_clusters = cdp.filter_clusters_by_features(clusters_list, params)

                result = defaultdict(set)
                for c in new_clusters:
                    if len(c) == 1:
                        index = c[0]
                        result[face_labels[index]].add(face_ids[index])
                for key in result.keys():
                    result[key] = list(result[key])
                # key为标签，val为过滤的face_id
                with open(cache_path, "w") as file:
                    for label in result:
                        pic_ids = result[label]
                        file.write("{}:{}\n".format(label, ",".join(pic_ids)))
            else:
                result = defaultdict(set)
        else:
            lines = open(cache_path, "r").readlines()
            result = {}
            for line in lines:
                label, pic_ids = line.strip().split(":")
                result[int(label)] = [p for p in pic_ids.split(",")]
        for label in list(result.keys()):
            # 已经不在了,就不要了
            if label not in self.label2pic or label in self.filter_labels:
                del result[label]
        return result

    def inter_check(self, th, check_count=10, leveldb_feature_path=os.path.expanduser("~/datasets/cacher/features")):
        cache_path = self.label_path + ".inter_check_%.2f" % th
        if not os.path.exists(cache_path):
            if os.path.exists(leveldb_feature_path):
                features = []
                face_ids = []
                face_labels = []
                clusters = defaultdict(list)
                fea_db = leveldb.LevelDB(leveldb_feature_path, max_open_files=100)
                for batch_index, label in enumerate(self.order_labels):
                    if batch_index % 1000 == 0:
                        logger.info("label_features batch_index/count %s/%s", batch_index, self.label_len)
                    pic_ids = self.label2pic[label]
                    pic_ids = random.sample(pic_ids, min(check_count, len(pic_ids)))
                    for index, pic_id in enumerate(pic_ids):
                        clusters[label].append(index)
                        face_ids.append(pic_id)
                        face_labels.append(label)
                        try:
                            pic_id = str(pic_id).encode('utf-8')
                            data = fea_db.Get(pic_id)
                            features.append(np.frombuffer(data, dtype=np.float32)[:512])
                        except Exception as e:
                            logger.info("pic_id %s no pic", pic_id)
                            sys.exit()

                face_features = np.array(features, dtype=np.float32)
                max_size = int(math.sqrt(len(face_features)) * 20)
                params = cdp.ClusterParams(features=face_features, labels=np.ones(len(face_features)) * -1, th_knn=th, max_size=max_size, single_filter=False)
                pred_labels, _ = cdp.cluster(params)

                label2faceids = defaultdict(list)
                label2labels = defaultdict(list)
                for index, label in enumerate(pred_labels):
                    if label == -1:
                        continue
                    label2faceids[label].append(face_ids[index])
                    label2labels[label].append(face_labels[index])

                conflict_list = []
                for key in label2labels:
                    label_face_ids = label2faceids[key]
                    label_list = label2labels[key]
                    # print(label_list)
                    if len(set(label_list)) > 1:
                        conflict_list.append([label_list, label_face_ids])
                # logger.info("label_inter_check conflict_list %s", conflict_list)
                # key为标签，val为过滤的face_id
                with open(cache_path, "w") as file:
                    for item in conflict_list:
                        label_list_str = ",".join([str(l) for l in item[0]])
                        face_ids_str = ",".join(item[1])
                        file.write("{};{}\n".format(label_list_str, face_ids_str))
            else:
                conflict_list = []
        else:
            with cost.Timer("load file"):
                lines = open(cache_path, "r").readlines()
                conflict_list = []
                for line in lines:
                    label_list_str, face_ids_str = line.strip().split(";")
                    label_list = [int(l) for l in label_list_str.split(",")]
                    label_face_ids = [face_id for face_id in face_ids_str.split(",")]
                    conflict_list.append([label_list, label_face_ids])

        with cost.Timer("filter"):
            conflict_list_filtered = []
            pic_ids_set = set(self.pic_ids)
            for item in conflict_list:
                labels = []
                pic_ids = []
                for index in range(len(item[0])):
                    label = item[0][index]
                    pic_id = item[1][index]
                    if label in self.label2pic and pic_id in pic_ids_set and label not in self.filter_labels and label not in self.replace_labels:
                        labels.append(label)
                        pic_ids.append(pic_id)
                if len(set(labels)) > 1:
                    conflict_list_filtered.append([labels, pic_ids])
        return conflict_list_filtered

    def merge_label(self, labels):
        labels = list(set(labels))
        labels = sorted(labels)
        if len(labels) > 1:
            assert len(self.replace_label_set.intersection(set(labels))) == 0
            self.replace_label_set.update(set(labels))
            for label in labels[:-1]:
                self.replace_labels[label] = labels[-1]
            merged_dict = defaultdict(set)
            for k, v in self.replace_labels.items():
                merged_dict[v].add(k)
                merged_dict[v].add(v)
            with open(self.path_label_merged, "w") as file:
                for target in merged_dict:
                    labels = merged_dict[target]
                    labels = sorted(labels)
                    labels = [str(l) for l in labels]
                    file.write((",").join(labels))
                    file.write("\n")

    def merge_all(self, th):
        cache_path = self.label_path + ".inter_check_%.2f" % th
        if not os.path.exists(cache_path):
            logger.info("merge_all cache_path %s not existed", cache_path)
            return
        with cost.Timer("load file"):
            lines = open(cache_path, "r").readlines()
            conflict_list = []
            for line in lines:
                label_list_str, face_ids_str = line.strip().split(";")
                label_list = [int(l) for l in label_list_str.split(",")]
                label_face_ids = [face_id for face_id in face_ids_str.split(",")]
                conflict_list.append([label_list, label_face_ids])
        with cost.Timer("filter"):
            pic_ids_set = set(self.pic_ids)
            for item in conflict_list:
                labels = []
                pic_ids = []
                for index in range(len(item[0])):
                    label = item[0][index]
                    pic_id = item[1][index]
                    if label in self.label2pic and pic_id in pic_ids_set and label not in self.filter_labels and label not in self.replace_labels:
                        labels.append(label)
                        pic_ids.append(pic_id)
                if len(set(labels)) > 1:
                    # assert len(self.replace_label_set.intersection(set(labels))) == 0
                    # self.replace_label_set.update(set(labels))
                    labels = sorted(labels)
                    for label in labels[:-1]:
                        self.replace_labels[label] = labels[-1]

        with cost.Timer("writer merged"):
            merged_dict = defaultdict(set)
            for k, v in self.replace_labels.items():
                if k != v and k in merged_dict:
                    # 已经当做目标了,就循环把原来的目标改为现在的
                    origin = merged_dict[k]
                    del merged_dict[k]
                    merged_dict[v].update(origin)
                    merged_dict[v].add(k)
                    merged_dict[v].add(v)
                    logger.info("merged passed %s->%s origin %s", k, v, origin)
                else:
                    merged_dict[v].add(k)
                    merged_dict[v].add(v)
            with open(self.path_label_merged, "w") as file:
                for target in merged_dict:
                    labels = merged_dict[target]
                    labels = sorted(labels)
                    labels = [str(l) for l in labels]
                    file.write((",").join(labels))
                    file.write("\n")

    def merge_filter(self, th):
        cache_path = self.label_path + ".inter_check_%.2f" % th
        if not os.path.exists(cache_path):
            logger.info("merge_filter cache_path %s not existed", cache_path)
            return
        with cost.Timer("load file"):
            lines = open(cache_path, "r").readlines()
            conflict_list = []
            for line in lines:
                label_list_str, face_ids_str = line.strip().split(";")
                label_list = [int(l) for l in label_list_str.split(",")]
                label_face_ids = [face_id for face_id in face_ids_str.split(",")]
                conflict_list.append([label_list, label_face_ids])
        with cost.Timer("filter"):
            filter_count = 0
            pic_ids_set = set(self.pic_ids)
            for item in conflict_list:
                labels = []
                pic_ids = []
                for index in range(len(item[0])):
                    label = item[0][index]
                    pic_id = item[1][index]
                    if label in self.label2pic and pic_id in pic_ids_set and label not in self.filter_labels and label not in self.replace_labels:
                        labels.append(label)
                        pic_ids.append(pic_id)
                if len(set(labels)) > 1:
                    max_label = None
                    max_count = 0
                    for label in labels:
                        if len(self.label2pic[label]) > max_count:
                            max_count = len(self.label2pic[label])
                            max_label = label
                    for label in labels:
                        if max_label != label:
                            filter_count += 1
                            self.filter_labels.add(label)
            logger.info("filter_count %s", filter_count)

        with open(self.path_imgidx_filter, "w") as file:
            for l in sorted(self.filter_labels, reverse=False):
                file.write(str(l))
                file.write("\n")

    def label_features(self):
        leveldb_feature_path = self.leveldb_feature_path
        if leveldb_feature_path is not None and os.path.exists(leveldb_feature_path):
            ret_features = nd.empty((self.label_len, 512))
            fea_db = leveldb.LevelDB(leveldb_feature_path, max_open_files=100)
            for batch_index, label in enumerate(self.order_labels):
                if batch_index % 1000 == 0:
                    logger.info("label_features batch_index/count %s/%s", batch_index, self.label_len)
                pic_id = self.label2pic[label][0]
                try:
                    pic_id = str(pic_id).encode('utf-8')
                    data = fea_db.Get(pic_id)
                    ret_features[batch_index][:] = np.frombuffer(data, dtype=np.float32)[:512] * 0.3
                except Exception as e:
                    logger.info("pic_id %s no pic", pic_id)
        else:
            ret_features = mx.nd.random.uniform(shape=(1, 512))
            ret_features = ret_features / ret_features.norm() * 0.2
        return ret_features

    def before_next_label(self, label):
        if label in self.order_labels:
            index = self.order_labels.index(label)
            last = 0 if index == 0 else index - 1
            next = index + 1 if index < len(self.order_labels) - 1 else index
            return self.order_labels[last], self.order_labels[next]
        return 0, 0

    def label_by_pic_id(self, pic_id):
        if pic_id in self.pic_ids:
            index = self.pic_ids.index(pic_id)
            return self.labels[index]
        return -1

    @property
    def pic_len(self):
        return len(self.pic_ids)

    @property
    def label_len(self):
        return len(self.label2pic)

    def __len__(self):
        return self.pic_len

    def __getitem__(self, idx):
        if idx < len(self.pic_ids):
            pic_id = self.pic_ids[idx]
            label = self.labels[idx]
            try:
                pic_id = str(pic_id).encode('utf-8')
                data = self.pic_db.Get(pic_id)
                img = mx.image.imdecode(data)
                return img, self.train_labels[label], pic_id
            except Exception as e:
                logger.info("pic_id %s no pic", pic_id)
        else:
            print("get_item error")
            assert False

    def binay_by_idx(self, pic_id):
        try:
            pic_id = str(pic_id).encode('utf-8')
            data = self.pic_db.Get(pic_id)
            return data
        except Exception as e:
            logger.info("pic_id %s no pic", pic_id)
            return None


class FaceImageIter(io.DataIter):
    def __init__(self, batch_size, data_shape, dataset, shuffle=False, gauss=0, data_name='data', label_name='softmax_label', queue_size=64):
        super(FaceImageIter, self).__init__()
        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.rand_mirror = False
        self.gauss = gauss
        self.image_size = '%d,%d' % (data_shape[1], data_shape[2])
        self.provide_label = [(label_name, (batch_size,))]
        # print(self.provide_label[0][1])

        self.dataset = dataset
        self.queue_size = queue_size
        self.running = True
        self.iter_start = False
        for i in range(1):
            self.thread = threading.Thread(target=self.process_data)
            self.thread.daemon = True
            self.thread.start()
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        logger.info('call reset() iter_start %s', self.iter_start)
        if self.iter_start:
            # 没有走完一个迭代不能重置
            return
        self.data_queue = Queue(self.queue_size)
        self.seq = Queue(len(self.dataset))
        self.dataset.reset()
        batch_count = int(len(self.dataset) / self.batch_size)
        if self.shuffle:
            indexes = list(range(len(self.dataset)))
            random.shuffle(indexes)
            for b in range(batch_count):
                self.seq.put(indexes[b * self.batch_size:(b + 1) * self.batch_size])
        else:
            indexes = list(range(len(self.dataset)))
            for b in range(batch_count):
                self.seq.put(indexes[b * self.batch_size:(b + 1) * self.batch_size])
        self.iter_start = True

    def num_samples(self):
        return self.dataset.pic_len

    def num_class(self):
        return self.dataset.label_len

    def process_data(self):
        while self.running:
            if self.iter_start:
                data = self.next_data()
                if data is None:
                    self.data_queue.put(None)
                    self.iter_start = False
                else:
                    self.data_queue.put(data)
            else:
                time.sleep(0.1)

    def next_data(self):
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        if self.seq.qsize() > 0:
            indexes = self.seq.get()
            for i, idx in enumerate(indexes):
                _data, label, pic_id = self.dataset[idx]
                # label, _data, bbox, landmark = self.seq.get()
                if _data.shape[0] != self.data_shape[1]:
                    _data = mx.image.resize_short(_data, self.data_shape[1])
                if self.gauss:
                    _rd = random.randint(0, 1)
                    if _rd == 1:
                        _data = cv2.GaussianBlur(_data.asnumpy(), (5, 5), 5)
                        _data = mx.ndarray.array(_data)
                if self.rand_mirror:
                    _rd = random.randint(0, 1)
                    if _rd == 1:
                        _data = mx.ndarray.flip(data=_data, axis=1)
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                for datum in data:
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
            return io.DataBatch([batch_data], [batch_label], batch_size - i)
        else:
            return None

    def print_info(self):
        logger.info("data_queue size %s", self.data_queue.qsize())

    def next(self):
        data = self.data_queue.get()
        # logger.info("data_queue size %s", self.data_queue.qsize())
        if data is None:
            raise StopIteration
        return data

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))


class ListDataset(mx.gluon.data.Dataset):
    def __init__(self, *args):
        super(ListDataset, self).__init__()
        self.dbs = args
        self.reset()

    @property
    def max_images(self):
        # 子集合暂不必重置
        return self.dbs[0].max_images

    @max_images.setter
    def max_images(self, val):
        # 子集合暂不必重置
        for index, d in enumerate(self.dbs):
            d.max_images = val

    def reset(self):
        # 子集合暂不必重置
        for index, d in enumerate(self.dbs):
            d.reset()
        self.data_len = 0
        self.label_len = 0
        self.steps = []
        self.glabal_label = 0
        self.alias_label = {}
        for index, d in enumerate(self.dbs):
            self.data_len += len(d)
            self.label_len += d.label_len
            self.steps.append(self.data_len)
            for l in range(d.label_len):
                self.alias_label[(index, l)] = self.glabal_label
                self.glabal_label += 1

    @property
    def pic_len(self):
        return self.data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        db = None
        db_index = 0
        last = 0
        for index, step in enumerate(self.steps):
            if idx < step:
                db = self.dbs[index]
                db_index = idx - last
                break
            last = step

        img, label_id, pic_id = db[db_index]
        alias = self.alias_label[(index, label_id)]
        return img, alias, pic_id

    def label_features(self):
        features = []
        for db in self.dbs:
            features.append(db.label_features())
        return mx.nd.concat(*features, dim=0)


if __name__ == '__main__':
    from PIL import Image

    random.seed(100)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

    leveldb_path = os.path.expanduser("~/datasets/cacher/pictures")
    label_path = os.path.expanduser("~/datasets/cacher/pictures.labels.40/pictures.labels.40.38")

    dataset = FaceDataset(leveldb_path, label_path, min_images=10, max_images=10)
    print(len(dataset))

    face = dataset[1100][0].asnumpy()
    im = Image.fromarray(face)
    if not os.path.exists("output"):
        os.mkdir("output")
    im.save("output/tmp1.jpg", quality=95)

    dataset.reset()
    face = dataset[1100][0].asnumpy()
    im = Image.fromarray(face)
    if not os.path.exists("output"):
        os.mkdir("output")
    im.save("output/tmp2.jpg", quality=95)
