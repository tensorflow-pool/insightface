import os

import mxnet as mx

data_shape = (3, 112, 112)
data_dir = os.path.expanduser("~/datasets/glintasia")
path_imgidx = os.path.join(data_dir, "train.idx")
path_imgrec = os.path.join(data_dir, "train.rec")

data_iter = mx.image.ImageIter(batch_size=64, data_shape=(3, 112, 112), label_width=2,
                               path_imgrec=path_imgrec,
                               path_imgidx=path_imgidx)

data_iter.reset()
batch = data_iter.next()
real_batch = batch.data[0]
print(real_batch.shape)