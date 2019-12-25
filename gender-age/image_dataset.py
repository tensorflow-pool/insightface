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
from queue import Queue

import cv2
import mxnet as mx
import numpy as np
from mxnet import io, nd

logger = logging.getLogger()


class FaceDataset(mx.gluon.data.Dataset):
    pic_db_dict = {}

    def __init__(self, leveldb_path, label_path):
        super(FaceDataset, self).__init__()
        assert leveldb_path
        logger.info('loading FaceDataset %s %s', leveldb_path, label_path)
        self.leveldb_path = leveldb_path
        self.label_path = label_path

        if leveldb_path in self.pic_db_dict:
            self.pic_db = self.pic_db_dict[leveldb_path]
        else:
            self.pic_db = leveldb.LevelDB(leveldb_path, max_open_files=100)
            self.pic_db_dict[leveldb_path] = self.pic_db
        with open(self.label_path, "r") as file:
            lines = file.readlines()
            self.pic_ids = []
            self.pic2agesex = {}
            for index, line in enumerate(lines):
                # if index > 3200:
                #     break
                pic_id, sex, age = line.strip().split(",")
                self.pic_ids.append(pic_id)
                sex = int(sex)
                age = int(age)
                self.pic2agesex[pic_id] = [sex, age]

        logger.info("final pic_ids %s labels %s", len(self.pic_ids), len(self.pic2agesex))

    @property
    def pic_len(self):
        return len(self.pic_ids)

    def __len__(self):
        return self.pic_len

    def __getitem__(self, idx):
        if idx < len(self.pic_ids):
            pic_id = self.pic_ids[idx]
            sex, age = self.pic2agesex[pic_id]
            try:
                pic_id = str(pic_id).encode('utf-8')
                data = self.pic_db.Get(pic_id)
                img = mx.image.imdecode(data)
                return img, sex, age, pic_id
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
    def __init__(self, batch_size, data_shape, dataset, shuffle=False, gauss=False, rand_mirror=False, data_name='data', label_name='softmax_label', queue_size=64):
        super(FaceImageIter, self).__init__()
        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.rand_mirror = rand_mirror
        self.gauss = gauss
        self.image_size = '%d,%d' % (data_shape[1], data_shape[2])
        self.provide_label = [(label_name, (batch_size, 101))]
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

    @property
    def pic_len(self):
        return self.dataset.pic_len

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
                _data, sex, age, pic_id = self.dataset[idx]
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
                    plabel = np.zeros(shape=(101,), dtype=np.float32)
                    plabel[0] = sex
                    if age == 0:
                        age = 1
                    if age > 100:
                        age = 100
                    plabel[1:age + 1] = 1
                    batch_label[i][:] = plabel
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


if __name__ == '__main__':
    from PIL import Image

    random.seed(100)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

    leveldb_path = os.path.expanduser("~/datasets/cacher/pictures")
    label_path = os.path.expanduser("~/datasets/cacher/pictures.high.labels.37/left_pictures.labels.37.35_36.processed.v30.sex_age")

    dataset = FaceDataset(leveldb_path, label_path)
    print(len(dataset))

    item = dataset[11100]
    face = item[0].asnumpy()
    sex = item[1]
    age = item[2]
    im = Image.fromarray(face)
    if not os.path.exists("output"):
        os.mkdir("output")
    im.save("output/tmp_{}_{}.jpg".format(sex, age), quality=95)
