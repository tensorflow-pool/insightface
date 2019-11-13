# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import math
import os
import sys

import mxnet as mx
import mxnet.optimizer as optimizer
import numpy as np

from image_iter import FaceImageIter

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import finception_resnet_v2
import fmobilenet
import fmobilenetv2
import fmobilefacenet
import fxception
import fdensenet
import fdpn
import fnasnet
import spherenet
import verification
import time
from mxboard import SummaryWriter

args = None


class AccMetric(mx.metric.EvalMetric):
    def __init__(self, real=False):
        self.real = real
        self.axis = 1
        super(AccMetric, self).__init__('real_acc' if real else 'acc', axis=self.axis, output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        # 1embedding
        # 2loss
        # 3real loss
        # 4real t
        # 5 extra loss
        self.count += 1

        if self.real:
            pred_label = preds[2]
        else:
            pred_label = preds[1]

        label = labels[0] - 1
        indexes = []
        for index, l in enumerate(label):
            if l != -1:
                indexes.append(index)
        if len(indexes) == 0:
            return

        label = label[indexes]
        pred_label = pred_label[indexes]
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()

        if label.ndim == 2:
            label = label[:, 0]
        label = label.asnumpy().astype('int32').flatten()
        assert label.shape == pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(indexes)


class LossMetric(mx.metric.EvalMetric):
    def __init__(self, real=False):
        self.real = real
        self.axis = 1
        super(LossMetric, self).__init__('real_loss' if real else 'loss', axis=self.axis, output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        # 1embedding
        # 2loss
        # 3real loss
        # 4real t
        # 5 extra loss
        self.count += 1
        label = labels[0] - 1
        if self.real:
            softmax_val = preds[2]
        else:
            softmax_val = preds[1]

        indexes = []
        for index, l in enumerate(label):
            if l != -1:
                indexes.append(index)
        if len(indexes) == 0:
            return
        label = label[indexes]
        softmax_val = softmax_val[indexes]
        one_hot = mx.ndarray.one_hot(mx.ndarray.array(label, ctx=mx.gpu()), depth=args.num_classes, on_value=1, off_value=0)
        loss = -mx.ndarray.broadcast_mul(one_hot, softmax_val.log()).sum().asnumpy()
        self.sum_metric += loss
        self.num_inst += len(indexes)
        # logging.info("loss real %s count %s loss %s", self.real, count, loss)


class ExtraLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(ExtraLossMetric, self).__init__("extra_loss")
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        # 1embedding
        # 2loss
        # 3real loss
        # 4real t
        # 5 extra loss
        self.count += 1
        cur_loss = preds[-1].sum().asnumpy()
        self.sum_metric += cur_loss

        label = labels[0]
        cout = mx.ndarray.where(label, mx.nd.zeros_like(label), mx.nd.ones_like(label)).sum().asnumpy()
        self.num_inst += cout
        # logging.info("loss cout %s ExtraLossMetric %s", cout, cur_loss)


class ThetaMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(ThetaMetric, self).__init__('theta', axis=self.axis, output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        # 1embedding
        # 2loss
        # 3real loss
        # 4real t
        # 5 extra loss
        theta = preds[3].asnumpy()

        label = labels[0] - 1
        indexes = []
        for index, l in enumerate(label):
            if l != -1:
                indexes.append(index)
        if len(indexes) == 0:
            return

        theta = theta[indexes]
        self.sum_metric += theta.sum()
        self.num_inst += len(indexes)


class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__(
            'lossvalue', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1.0
        gt_label = preds[-2].asnumpy()
        # print(gt_label)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    # parser.add_argument('--data-dir', default='~/datasets/face_umd/faces_umd', help='training set directory')
    # parser.add_argument('--data-dir', default='~/datasets/ms1m-v1/faces_ms1m_112x112', help='training set directory')
    # parser.add_argument('--data-dir', default='~/datasets/glintasia', help='training set directory')
    # parser.add_argument('--data-dir', default='~/datasets/maysa', help='training set directory')

    # parser.add_argument('--data-dir', default='~/datasets/ms1m-retina', help='training set directory')
    # parser.add_argument('--rec', default='train_65_70_maysa_0.6_10_filtered_merged.rec', help='training set directory')

    # parser.add_argument('--data-dir', default='~/datasets/maysa', help='training set directory')
    # parser.add_argument('--rec', default='project_xm_huafu_5573k_q95_retina_pred_0.6_10_filtered_merged.rec', help='training set directory')

    parser.add_argument('--data-dir', default='~/datasets/maysa', help='training set directory')
    parser.add_argument('--rec', default='glint_maysa_0.5_10_300.rec', help='training set directory')

    parser.add_argument('--lr', type=float, default=0.05, help='start learning rate')
    parser.add_argument('--target', type=str, default='lfw', help='verification targets')
    parser.add_argument('--per-batch-size', type=int, default=48, help='batch size in each context')

    parser.add_argument('--prefix', default='../model-output', help='directory to save model.')
    # parser.add_argument('--pretrained', default='../models/model-r100-ii-1-16/model,29', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='../models/model-r34-amf/model,0', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='../models/model-r34-7-19/model,172000', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='../models/r100-iccv/model,1', help='pretrained model to load')
    parser.add_argument('--pretrained', default='~/models/models_retina100_2019-10-18/model,486201', help='pretrained model to load')
    parser.add_argument('--loss-type', type=int, default=4, help='loss type 5的时候为cos(margin_a*θ+margin_m) - margin_b;cos(θ+0.3)-0.2 or cos(θ+0.5)')
    parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
    parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
    parser.add_argument('--network', default='r100', help='specify network')
    parser.add_argument('--image-size', default='112,112', help='specify input image height and width')
    parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
    parser.add_argument('--version-input', type=int, default=1, help='network input config 1代表第一次卷积7x7-2改为3x3-1')
    parser.add_argument('--version-output', type=str, default='E', help='network embedding output config e代表的是bn-drop-fc-bn结构')
    parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config 3代表的是arc论文中对残差网络单元的修改，增加了更多bn和prelu')
    parser.add_argument('--version-multiplier', type=float, default=1.0, help='filters multiplier')
    parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
    parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')
    parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
    parser.add_argument('--fc7-lr-mult', type=float, default=1.0, help='lr mult for fc7')
    parser.add_argument("--fc7-no-bias", default=False, action="store_true", help="fc7 no bias flag")
    parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss,')
    parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
    parser.add_argument('--margin-a', type=float, default=1.0, help='')
    parser.add_argument('--margin-b', type=float, default=0.2, help='')
    parser.add_argument('--easy-margin', type=int, default=0, help='')
    parser.add_argument('--margin', type=int, default=4, help='margin for sphere')
    parser.add_argument('--beta', type=float, default=1000., help='param for sphere')
    parser.add_argument('--beta-min', type=float, default=5., help='param for sphere')
    parser.add_argument('--beta-freeze', type=int, default=0, help='param for sphere')
    parser.add_argument('--gamma', type=float, default=0.12, help='param for sphere')
    parser.add_argument('--power', type=float, default=1.0, help='param for sphere')
    parser.add_argument('--scale', type=float, default=0.9993, help='param for sphere')
    parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
    parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
    parser.add_argument('--color', type=int, default=0, help='color jittering aug')
    parser.add_argument('--images-filter', type=int, default=0, help='minimum images per identity filter')
    parser.add_argument('--ce-loss', default=False, action='store_true', help='if output ce loss')
    args = parser.parse_args()
    return args


def get_symbol(args, arg_params, aux_params):
    data_shape = (args.image_channel, args.image_h, args.image_w)
    image_shape = ",".join([str(x) for x in data_shape])
    margin_symbols = []
    if args.network[0] == 'd':
        embedding = fdensenet.get_symbol(args.emb_size, args.num_layers,
                                         version_se=args.version_se, version_input=args.version_input,
                                         version_output=args.version_output, version_unit=args.version_unit)
    elif args.network[0] == 'm':
        print('init mobilenet', args.num_layers)
        if args.num_layers == 1:
            embedding = fmobilenet.get_symbol(args.emb_size,
                                              version_input=args.version_input,
                                              version_output=args.version_output,
                                              version_multiplier=args.version_multiplier)
        else:
            embedding = fmobilenetv2.get_symbol(args.emb_size)
    elif args.network[0] == 'i':
        print('init inception-resnet-v2', args.num_layers)
        embedding = finception_resnet_v2.get_symbol(args.emb_size,
                                                    version_se=args.version_se, version_input=args.version_input,
                                                    version_output=args.version_output, version_unit=args.version_unit)
    elif args.network[0] == 'x':
        print('init xception', args.num_layers)
        embedding = fxception.get_symbol(args.emb_size,
                                         version_se=args.version_se, version_input=args.version_input,
                                         version_output=args.version_output, version_unit=args.version_unit)
    elif args.network[0] == 'p':
        print('init dpn', args.num_layers)
        embedding = fdpn.get_symbol(args.emb_size, args.num_layers,
                                    version_se=args.version_se, version_input=args.version_input,
                                    version_output=args.version_output, version_unit=args.version_unit)
    elif args.network[0] == 'n':
        print('init nasnet', args.num_layers)
        embedding = fnasnet.get_symbol(args.emb_size)
    elif args.network[0] == 's':
        print('init spherenet', args.num_layers)
        embedding = spherenet.get_symbol(args.emb_size, args.num_layers)
    elif args.network[0] == 'y':
        print('init mobilefacenet', args.num_layers)
        embedding = fmobilefacenet.get_symbol(args.emb_size, bn_mom=args.bn_mom, version_output=args.version_output)
    else:
        print('init resnet', args.num_layers)
        embedding = fresnet.get_symbol(args.emb_size, args.num_layers,
                                       version_se=args.version_se, version_input=args.version_input,
                                       version_output=args.version_output, version_unit=args.version_unit,
                                       version_act=args.version_act)
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label - 1
    base_loss = mx.sym.where(all_label, mx.sym.ones_like(all_label), mx.symbol.zeros_like(all_label))
    extra_loss = mx.sym.where(all_label, mx.sym.zeros_like(all_label), mx.symbol.ones_like(all_label))
    _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=args.fc7_lr_mult,
                                 wd_mult=args.fc7_wd_mult)
    if args.loss_type == 0:  # softmax
        if args.fc7_no_bias:
            fc7 = mx.sym.FullyConnected(data=embedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                        name='fc7')
        else:
            _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
            fc7 = mx.sym.FullyConnected(data=embedding, weight=_weight, bias=_bias, num_hidden=args.num_classes,
                                        name='fc7')
    elif args.loss_type == 1:  # sphere
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes,
                              weight=_weight,
                              beta=args.beta, margin=args.margin, scale=args.scale,
                              beta_min=args.beta_min, verbose=2000, name='fc7')
    elif args.loss_type == 2:  # cos
        s = args.margin_s
        m = args.margin_m
        assert (s > 0.0)
        assert (m > 0.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        s_m = s * m
        gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=s_m, off_value=0.0)
        fc7 = fc7 - gt_one_hot
    elif args.loss_type == 4:  # arc
        s = args.margin_s  # 64
        m = args.margin_m  # 0.5
        assert s > 0.0
        assert m >= 0.0
        assert m < (math.pi / 2)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        origin_fc7 = fc7
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy / s
        # 避免ｎａｎ出现
        cal_cos_t = mx.sym.where(cos_t > 1, mx.sym.ones_like(cos_t), cos_t)
        cal_cos_t = mx.sym.where(cal_cos_t < -1, -mx.sym.ones_like(cos_t), cal_cos_t)
        origin_t = mx.sym.degrees(mx.sym.arccos(cal_cos_t))
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi - m) * m
        # threshold = 0.0
        threshold = math.cos(math.pi - m)
        if args.easy_margin:
            cond = mx.symbol.Activation(data=cos_t, act_type='relu')
        else:
            cond_v = cos_t - threshold
            cond = mx.symbol.Activation(data=cond_v, act_type='relu')
        body = cos_t * cos_t
        body = 1.0 - body
        sin_t = mx.sym.sqrt(body)
        new_zy = cos_t * cos_m
        b = sin_t * sin_m
        new_zy = new_zy - b
        new_zy = new_zy * s
        if args.easy_margin:
            zy_keep = zy
        else:
            zy_keep = zy - s * mm
        new_zy = mx.sym.where(cond, new_zy, zy_keep)

        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7 + body
    elif args.loss_type == 5:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        origin_fc7 = fc7
        if args.margin_a != 1.0 or args.margin_m != 0.0 or args.margin_b != 0.0:
            if args.margin_a == 1.0 and args.margin_m == 0.0:
                s_m = s * args.margin_b
                gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=s_m, off_value=0.0)
                fc7 = fc7 - gt_one_hot
            else:
                zy = mx.sym.pick(fc7, gt_label, axis=1)
                cos_t = zy / s
                t = mx.sym.degrees(mx.sym.arccos(cos_t))
                origin_t = t
                if args.margin_a != 1.0:
                    t = t * args.margin_a
                if args.margin_m > 0.0:
                    t = t + args.margin_m
                body = mx.sym.cos(t)
                if args.margin_b > 0.0:
                    body = body - args.margin_b
                new_zy = body * s
                diff = new_zy - zy
                diff = mx.sym.expand_dims(diff, 1)
                gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
                body = mx.sym.broadcast_mul(gt_one_hot, diff)
                fc7 = fc7 + body
    elif args.loss_type == 6:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        assert args.margin_b > 0.0
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy / s
        t = mx.sym.arccos(cos_t)
        intra_loss = t / np.pi
        intra_loss = mx.sym.mean(intra_loss)
        # intra_loss = mx.sym.exp(cos_t*-1.0)
        intra_loss = mx.sym.MakeLoss(intra_loss, name='intra_loss', grad_scale=args.margin_b)
        if m > 0.0:
            t = t + m
            body = mx.sym.cos(t)
            new_zy = body * s
            diff = new_zy - zy
            diff = mx.sym.expand_dims(diff, 1)
            gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
            body = mx.sym.broadcast_mul(gt_one_hot, diff)
            fc7 = fc7 + body
    elif args.loss_type == 7:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        assert args.margin_b > 0.0
        assert args.margin_a > 0.0
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                    name='fc7')
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy / s
        t = mx.sym.arccos(cos_t)

        # counter_weight = mx.sym.take(_weight, gt_label, axis=1)
        # counter_cos = mx.sym.dot(counter_weight, _weight, transpose_a=True)
        counter_weight = mx.sym.take(_weight, gt_label, axis=0)
        counter_cos = mx.sym.dot(counter_weight, _weight, transpose_b=True)
        # counter_cos = mx.sym.minimum(counter_cos, 1.0)
        # counter_angle = mx.sym.arccos(counter_cos)
        # counter_angle = counter_angle * -1.0
        # counter_angle = counter_angle/np.pi #[0,1]
        # inter_loss = mx.sym.exp(counter_angle)

        # counter_cos = mx.sym.dot(_weight, _weight, transpose_b=True)
        # counter_cos = mx.sym.minimum(counter_cos, 1.0)
        # counter_angle = mx.sym.arccos(counter_cos)
        # counter_angle = mx.sym.sort(counter_angle, axis=1)
        # counter_angle = mx.sym.slice_axis(counter_angle, axis=1, begin=0,end=int(args.margin_a))

        # inter_loss = counter_angle*-1.0 # [-1,0]
        # inter_loss = inter_loss+1.0 # [0,1]
        inter_loss = counter_cos
        inter_loss = mx.sym.mean(inter_loss)
        inter_loss = mx.sym.MakeLoss(inter_loss, name='inter_loss', grad_scale=args.margin_b)
        if m > 0.0:
            t = t + m
            body = mx.sym.cos(t)
            new_zy = body * s
            diff = new_zy - zy
            diff = mx.sym.expand_dims(diff, 1)
            gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
            body = mx.sym.broadcast_mul(gt_one_hot, diff)
            fc7 = fc7 + body
    # 1embedding
    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label=gt_label, name='softmax', normalization='valid')
    base_loss_expanded = mx.sym.expand_dims(base_loss, -1)
    base_loss_val = mx.sym.make_loss(mx.sym.broadcast_mul(base_loss_expanded, softmax), name="base_loss")
    # 2loss
    out_list.append(base_loss_val)
    # 3real loss
    origin_softmax_fc7 = mx.symbol.softmax(origin_fc7)
    out_list.append(mx.symbol.BlockGrad(mx.sym.broadcast_mul(base_loss_expanded, origin_softmax_fc7)))
    # out_list.append(mx.symbol.BlockGrad(mx.symbol.softmax(fc7)))
    # origin_softmax_fc7 = mx.symbol.softmax(origin_fc7)
    # out_list.append(mx.symbol.BlockGrad(origin_softmax_fc7))
    # 4real t
    # out_list.append(mx.symbol.BlockGrad(mx.sym.broadcast_mul(base_loss, origin_t)))
    # 不要归零，看看iqiyi loss 是否有效
    out_list.append(mx.symbol.BlockGrad(origin_t))
    if args.loss_type == 6:
        out_list.append(intra_loss)
    if args.loss_type == 7:
        out_list.append(inter_loss)
        # out_list.append(mx.sym.BlockGrad(counter_weight))
        # out_list.append(intra_loss)
    if args.ce_loss:
        # ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
        body = mx.symbol.SoftmaxActivation(data=fc7)
        body = mx.symbol.log(body)
        _label = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=-1.0, off_value=0.0)
        body = body * _label
        ce_loss = mx.symbol.sum(body) / args.per_batch_size
        out_list.append(mx.symbol.BlockGrad(ce_loss))
    extra_loss_val = -mx.sym.sum(mx.sym.log(origin_softmax_fc7), axis=-1)
    # 5 extra loss
    out_list.append(mx.sym.make_loss(0.00002 * extra_loss * extra_loss_val, name="extra_loss"))
    out = mx.symbol.Group(out_list)
    #
    return (out, arg_params, aux_params)


def train_net(args):
    prefix = time.strftime("%Y-%m-%d-%H:%M:%S")
    file_path = "train/models_{}".format(prefix)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    sw = SummaryWriter(logdir=file_path)
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler("{}/train.log".format(file_path))
    # create formatter#
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # add formatter to ch
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    prefix = file_path

    args.data_dir = os.path.expanduser(args.data_dir)
    args.pretrained = os.path.expanduser(args.pretrained)

    ctx = [mx.gpu()]
    logging.info('use gpu0')

    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    logging.info('num_layers %s', args.num_layers)
    if args.per_batch_size == 0:
        args.per_batch_size = 128
    args.batch_size = args.per_batch_size * args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3

    os.environ['BETA'] = str(args.beta)
    image_size = [int(x) for x in args.image_size.split(',')]
    assert len(image_size) == 2
    assert image_size[0] == image_size[1]
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    logging.info('image_size %s', image_size)

    path_imgrec = os.path.join(args.data_dir, args.rec)
    if args.loss_type == 1 and args.num_classes > 20000:
        args.beta_freeze = 5000
        args.gamma = 0.06

    data_shape = (args.image_channel, image_size[0], image_size[1])
    mean = None
    train_dataiter = FaceImageIter(
        batch_size=args.batch_size,
        data_shape=data_shape,
        path_imgrec=path_imgrec,
        shuffle=True,
        rand_mirror=args.rand_mirror,
        mean=mean,
        cutoff=args.cutoff,
        color_jittering=args.color,
        images_filter=args.images_filter,
    )
    args.num_classes = train_dataiter.num_class() - 1
    assert (args.num_classes > 0)
    logging.info('num_classes %s', args.num_classes)
    logging.info('Called with argument: %s', args)

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained) == 0:
        arg_params = None
        aux_params = None
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
        if args.network[0] == 's':
            data_shape_dict = {'data': (args.per_batch_size,) + data_shape}
            spherenet.init_weights(sym, data_shape_dict, args.num_layers)
    else:
        vec = args.pretrained.split(',')
        logging.info('loading %s', vec)
        sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
        arg_params['fc7_weight'] = arg_params['fc7_weight'][-args.num_classes:, :]
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    # label_name = 'softmax_label'
    # label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context=ctx,
        symbol=sym,
    )
    val_dataiter = None

    eval_metrics = [mx.metric.create([AccMetric(), AccMetric(True), LossMetric(), LossMetric(True), ExtraLossMetric(), ThetaMetric()])]
    # eval_metrics = []
    if args.ce_loss:
        metric2 = LossValueMetric()
        eval_metrics.append(mx.metric.create(metric2))

    if args.network[0] == 'r' or args.network[0] == 'y':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)  # resnet style
    elif args.network[0] == 'i' or args.network[0] == 'x':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)  # inception
    else:
        initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    # initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    _rescale = 1.0 / args.ctx_num / args.batch_size
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 20
    _cb = mx.callback.Speedometer(args.batch_size, som, auto_reset=True)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(args.data_dir, name + ".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            logging.info('ver %s', name)
        else:
            logging.info("path %s not existed", path)

    def ver_test(nbatch):
        results = []
        for i in range(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10,
                                                                               None, None)
            logging.info('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            # logging.info('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
        return results

    highest_acc = [0.0, 0.0]  # lfw and target
    # for i in range(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    if len(args.lr_steps) == 0:
        lr_steps = [8, 12, 16]
        lr_steps = [3, 6, 8]
        # if args.loss_type >= 1 and args.loss_type <= 7:
        #     lr_steps = [100000, 140000, 160000]
        p = train_dataiter.num_samples() / args.batch_size
        # 加速
        # if p > 20000:
        #     p = p / 2
        for l in range(len(lr_steps)):
            # lr_steps[l] = int(lr_steps[l])
            lr_steps[l] = int(lr_steps[l] * p)
        args.max_steps = 2 * lr_steps[-1] - lr_steps[-2]
    else:
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
    epoch_size = int(train_dataiter.num_samples() / args.batch_size)
    logging.info('lr_steps %s epoch_size %s', lr_steps, epoch_size)

    def _batch_callback(param):
        # global global_step
        global_step[0] += 1
        mbatch = global_step[0]
        for _lr in lr_steps:
            if mbatch == args.beta_freeze + _lr:
                opt.lr *= 0.1
                logging.info('lr change to %s', opt.lr)
                break

        acc = param.eval_metric.get_name_value()[0][1]
        loss = param.eval_metric.get_name_value()[1][1]
        real_acc = param.eval_metric.get_name_value()[2][1]
        real_loss = param.eval_metric.get_name_value()[3][1]
        if mbatch % 100 == 0:
            logging.info('lr-batch-epoch: lr %s, nbatch %s, epoch %s, step %s', opt.lr, param.nbatch, param.epoch, global_step[0])

        sw.add_scalar(tag='lr', value=opt.lr, global_step=mbatch)
        sw.add_scalar(tag='acc', value=acc, global_step=mbatch)
        sw.add_scalar(tag='loss', value=loss, global_step=mbatch)
        sw.add_scalar(tag='real_acc', value=real_acc, global_step=mbatch)
        sw.add_scalar(tag='real_loss', value=real_loss, global_step=mbatch)
        theta = model.get_outputs()[3].asnumpy()
        sw.add_histogram(tag="theta", values=theta, global_step=mbatch, bins=100)
        # logging.info("theta %s", theta)
        # logging.info('nbatch %s, epoch %s, step %s acc %s loss %s real_acc %s real_loss %s theta %s',
        #              param.nbatch, param.epoch, global_step[0], acc, loss, real_acc, real_loss, theta.mean())

        _cb(param)

        if mbatch % epoch_size == 0:
            if len(ver_list) > 0:
                acc_list = ver_test(mbatch)
                logging.info('[%d]Accuracy-Highest: %s' % (mbatch, acc_list))
                sw.add_scalar(tag='val', value=acc_list[0], global_step=global_step[0])

            logging.info('saving %s', mbatch)
            arg, aux = model.get_params()
            new_arg = {}
            for k in arg:
                if k == "fc7_weight":
                    continue
                new_arg[k] = arg[k]
            new_arg = arg
            mx.model.save_checkpoint(prefix + "/model", mbatch, model.symbol[0].get_children(), new_arg, aux)

        if mbatch <= args.beta_freeze:
            _beta = args.beta
        else:
            move = max(0, mbatch - args.beta_freeze)
            _beta = max(args.beta_min, args.beta * math.pow(1 + args.gamma * move, -1.0 * args.power))
        # logging.info('beta', _beta)
        os.environ['BETA'] = str(_beta)
        if args.max_steps > 0 and mbatch > args.max_steps:
            sys.exit(0)

    def epoch_cb(epoch, symbol, arg, aux):
        pass

    train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    model.fit(train_dataiter,
              begin_epoch=begin_epoch,
              num_epoch=end_epoch,
              eval_data=val_dataiter,
              eval_metric=eval_metrics,
              kvstore='device',
              optimizer=opt,
              # optimizer_params   = optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              batch_end_callback=_batch_callback,
              epoch_end_callback=epoch_cb)


def main():
    import gluoncv
    gluoncv.utils.random.seed(1)
    import random
    random.seed(1)
    # time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
