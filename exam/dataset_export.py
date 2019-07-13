import numbers

import mxnet as mx
from PIL import Image
from mxnet import recordio

# path_imgidx = "/home/lijc08/datasets/glintasia/train.idx"
# path_imgrec = "/home/lijc08/datasets/glintasia/train.rec"
path_imgidx = "/home/lijc08/datasets/glint/train.idx"
path_imgrec = "/home/lijc08/datasets/glint/train.rec"
imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
s = imgrec.read_idx(0)
header, _ = recordio.unpack(s)
if header.flag > 0:
    print('header0 label', header.label)
    header0 = (int(header.label[0]), int(header.label[1]))
    seq_identity = range(int(header.label[0]), 2830155)
    # seq_identity = range(int(header.label[0]), int(header.label[1]))

    id2range = {}
    imgidx = []
    for identity in seq_identity:
        s = imgrec.read_idx(identity)
        header, _ = recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        count = b - a
        id2range[identity] = (a, b)
        imgidx += range(a, b)
    print('id2range', id2range)
    print('imgidx', imgidx)


    def export(idx):
        s = imgrec.read_idx(idx)
        header, img = recordio.unpack(s)
        decodeImg = mx.image.imdecode(img)
        print("header", header)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = int(label[0])

        img = Image.fromarray(decodeImg.asnumpy(), 'RGB')
        img.save("glint/" + str(idx) + "_" + str(label) + ".jpeg", format='JPEG')


    # for idx in range(1, 2800000, 10000):
    for idx in range(1, 6600000, 10000):
        export(idx)
        export(idx + 1)
