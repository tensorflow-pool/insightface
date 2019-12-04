# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import leveldb
import logging
import os
import random
from collections import defaultdict

import cv2
import mxnet as mx
from mxnet import io, nd

logger = logging.getLogger()


class FaceDataset(mx.gluon.data.Dataset):
    pic_db_dict = {}

    def __init__(self, leveldb_path, label_path, min_images=0, max_images=11111111111, ignore_labels=set()):
        super(FaceDataset, self).__init__()
        assert leveldb_path
        logger.info('loading FaceDataset %s %s min_images %s max_images %s', leveldb_path, label_path, min_images, max_images)
        self.leveldb_path = leveldb_path
        self.label_path = label_path
        self.path_imgidx_filter = label_path + ".filter"
        self.filter_labels = set()
        if os.path.exists(self.path_imgidx_filter):
            filtered_labels = open(self.path_imgidx_filter).readlines()
            self.filter_labels = [int(l) for l in filtered_labels]
            self.filter_labels = set(self.filter_labels)

        if leveldb_path in self.pic_db_dict:
            self.pic_db = self.pic_db_dict[leveldb_path]
        else:
            self.pic_db = leveldb.LevelDB(leveldb_path, max_open_files=100)
            self.pic_db_dict[leveldb_path] = self.pic_db
        with open(self.label_path, "r") as file:
            lines = file.readlines()
            self.pic_ids = []
            self.labels = []
            self.label2pic = defaultdict(list)
            for index, line in enumerate(lines):
                pic_id, label = line.strip().split(",")
                label = int(label)
                if label != -1 or label in ignore_labels or label in self.filter_labels:
                    self.pic_ids.append(pic_id)
                    self.labels.append(label)
                    self.label2pic[label].append(pic_id)
        logger.info("origin pic_ids %s labels %s", len(self.pic_ids), len(self.label2pic))
        if min_images > 0:
            ignore_pic_ids = set()
            new_label2pic = defaultdict(list)
            for l in self.label2pic:
                c = self.label2pic[l]
                if len(c) >= min_images:
                    new_label2pic[l] = c[:max_images]
                    for ig in c[max_images:]:
                        ignore_pic_ids.add(ig)
            new_pic_ids = []
            new_labels = []
            for index in range(len(self.labels)):
                label = self.labels[index]
                if label in new_label2pic:
                    if self.pic_ids[index] not in ignore_pic_ids:
                        new_pic_ids.append(self.pic_ids[index])
                        new_labels.append(self.labels[index])
            self.pic_ids = new_pic_ids
            self.labels = new_labels
            self.label2pic = new_label2pic

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

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

    leveldb_path = os.path.expanduser("~/datasets/cacher/pictures")
    label_path = os.path.expanduser("~/datasets/cacher/pictures.labels.40/pictures.labels.40.38")

    dataset = FaceDataset(leveldb_path, label_path, min_images=0)
    print(len(dataset))
    face = dataset[1100][0].asnumpy()
    im = Image.fromarray(face)
    if not os.path.exists("output"):
        os.mkdir("output")
    im.save("output/tmp.jpg", quality=95)
