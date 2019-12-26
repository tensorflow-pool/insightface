from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import mxnet as mx
import numpy as np


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, args):
        self.args = args
        if args.gpu >= 0:
            ctx = mx.gpu(args.gpu)
        else:
            ctx = mx.cpu()
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        if len(args.model) > 0:
            self.model = get_model(ctx, image_size, args.model, 'fc1')
        # self.det_factor = 0.9
        self.image_size = image_size

    def get_ga(self, face_imgs):
        batch_imags = []
        for face_img in face_imgs:
            nimg = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2, 0, 1))
            batch_imags.append(aligned)
        data = mx.nd.array(batch_imags)
        data = mx.io.DataBatch(data=(data,))

        self.model.forward(data, is_train=False)
        ret = self.model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2]
        genders = np.argmax(g, axis=-1)
        a = ret[:, 2:202].reshape((-1, 100, 2))
        a = np.argmax(a, axis=-1)
        ages = a.sum(axis=-1)
        return genders, ages
