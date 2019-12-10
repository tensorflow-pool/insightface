# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import leveldb
import logging
import os
import random
import threading
from collections import defaultdict
from queue import Queue

import cv2
import mxnet as mx
import numpy as np
from mxnet import io, nd

logger = logging.getLogger()


class FaceDataset(mx.gluon.data.Dataset):
    pic_db_dict = {}

    def __init__(self, leveldb_path, label_path, pic_ignore=os.path.expanduser("~/datasets/cacher/picture.ignore"), min_images=0, max_images=11111111111, ignore_labels=set()):
        super(FaceDataset, self).__init__()
        assert leveldb_path
        logger.info('loading FaceDataset %s %s min_images %s max_images %s', leveldb_path, label_path, min_images, max_images)
        self.leveldb_path = leveldb_path
        self.label_path = label_path
        self.path_imgidx_filter = label_path + ".filter"
        self.path_label_check = label_path + ".check"
        self.filter_labels = set()
        if os.path.exists(self.path_imgidx_filter):
            filtered_labels = open(self.path_imgidx_filter).readlines()
            self.filter_labels = [int(l) for l in filtered_labels]
            self.filter_labels = set(self.filter_labels)
            # logger.info("FaceDataset filter_labels %s", self.filter_labels)

        self.ignore_pic_ids = set()
        if os.path.exists(pic_ignore):
            pic_ids = open(pic_ignore).readlines()
            pic_ids = [pic_id.strip() for pic_id in pic_ids]
            self.ignore_pic_ids = set(pic_ids)
            # logger.info("FaceDataset ignore_pic_ids %s", self.ignore_pic_ids)

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

    def check_warning_labels(self, leveldb_feature_path, th=None, random_select=30):
        if not os.path.exists(self.path_label_check):
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

            with open(self.path_label_check, "w") as file:
                items = list(label2score.items())
                items = sorted(items, key=lambda x: x[1])
                for k, v in items:
                    file.write(str(k) + "," + str(v))
                    file.write("\n")

        label2score = {}
        with open(self.path_label_check, "r") as file:
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

    def label_features(self, leveldb_feature_path):
        ret_features = nd.empty((self.label_len, 512))
        if os.path.exists(leveldb_feature_path):
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
    def __init__(self, batch_size, data_shape, dataset, shuffle=False, gauss=0, data_name='data', label_name='softmax_label'):
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
        self.seq = list(range(len(dataset)))
        self.cur = 0
        self.nbatch = 0
        self.is_init = False

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.dataset.reset()
        self.seq = list(range(len(self.dataset)))
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.seq)

    def num_samples(self):
        return self.dataset.pic_len

    def num_class(self):
        return self.dataset.label_len

    def next_sample(self):
        """Helper function for reading in next sample."""
        # set total batch size, for example, 1800, and maximum size for each people, for example 45
        while True:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            img, label, pic_id = self.dataset[idx]
            # logger.info("idx %s label %s", idx, label)
            return label, img, None, None

    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
        self.nbatch += 1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, _data, bbox, landmark = self.next_sample()
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
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    # print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i < batch_size:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

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
