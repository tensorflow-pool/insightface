import numbers
import os

import mxnet as mx
from PIL import Image
from mxnet import recordio

# path_imgidx = "/home/lijc08/datasets/glintasia/train.idx"
# path_imgrec = "/home/lijc08/datasets/glintasia/train.rec"
# output="glintasia"
# path_imgidx = "/home/lijc08/datasets/glint/train.idx"
# path_imgrec = "/home/lijc08/datasets/glint/train.rec"
# output="glint"
# path_imgidx = "/home/lijc08/datasets/ms1m-v1/faces_ms1m_112x112/train.idx"
# path_imgrec = "/home/lijc08/datasets/ms1m-v1/faces_ms1m_112x112/train.rec"
# output = "ms1m-v1"
# path_imgidx = "/home/lijc08/datasets/ms1m-v2/faces_emore/train.idx"
# path_imgrec = "/home/lijc08/datasets/ms1m-v2/faces_emore/train.rec"
# output = "ms1m-v2"
path_imgidx = "/home/lijc08/datasets/face_umd/faces_umd/train.idx"
path_imgrec = "/home/lijc08/datasets/face_umd/faces_umd/train.rec"
output = "face_umd"

if not os.path.exists(output):
    os.mkdir(output)
imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
s = imgrec.read_idx(0)
header, _ = recordio.unpack(s)
if header.flag > 0:
    print('header0 label', header.label)


    def export(idx):
        s = imgrec.read_idx(idx)
        header, img = recordio.unpack(s)
        decodeImg = mx.image.imdecode(img)
        print("header", header)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = int(label[0])

        img = Image.fromarray(decodeImg.asnumpy(), 'RGB')
        img.save(output + "/" + str(idx) + "_" + str(label) + ".jpeg", format='JPEG')


    # for idx in range(1, 2800000, 10000):
    # for idx in range(1, 6600000, 10000):
    for idx in range(1, int(header.label[0]), 10000):
        export(idx)
        export(idx + 1)
