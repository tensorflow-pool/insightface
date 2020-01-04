# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import math
import os
import sys
from collections import defaultdict

import mxnet as mx
import mxnet.optimizer as optimizer
import numpy as np

from image_dataset import FaceImageIter, FaceDataset, ListDataset

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import git

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

cos_ts = []


class AccMetric(mx.metric.EvalMetric):
    def __init__(self, real=False):
        self.real = real
        self.axis = 1
        super(AccMetric, self).__init__('real_acc' if real else 'acc', axis=self.axis, output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count += 1
        label = labels[0]
        if self.real:
            pred_label = preds[2]
        else:
            pred_label = preds[1]
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim == 2:
            label = label[:, 0]
        label = label.astype('int32').flatten()
        assert label.shape == pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)


class LossMetric(mx.metric.EvalMetric):
    def __init__(self, real=False):
        self.real = real
        self.axis = 1
        super(LossMetric, self).__init__('real_loss' if real else 'loss', axis=self.axis, output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count += 1
        if self.real:
            softmax_val = preds[2]
        else:
            softmax_val = preds[1]
        loss = -mx.ndarray.broadcast_mul(mx.ndarray.one_hot(mx.ndarray.array(labels[0], ctx=mx.gpu()), depth=args.num_classes, on_value=1, off_value=0), softmax_val.log()).sum(
            axis=1).mean()
        # logging.info("loss %s", loss)
        self.sum_metric += loss.asnumpy()
        self.num_inst += 1


class ThetaMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(ThetaMetric, self).__init__('theta', axis=self.axis, output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        theta = preds[3].asnumpy()
        self.sum_metric += theta.mean()
        self.num_inst += 1.0


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


# TrainParams = namedtuple('TrainParams', ['base_lr_steps', "lr_steps", "mas", "epoch_sizes"])
# TrainParams.__new__.__defaults__ = ([], [], [])

def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')

    # leveldb_path = os.path.expanduser("~/datasets/cacher/pictures")
    leveldb_path = os.path.expanduser("/opt/cacher/faces_webface_112x112")
    parser.add_argument('--leveldb_path', default=leveldb_path, help='training set directory')
    # label_path = os.path.expanduser("~/datasets/cacher/pictures.labels.40/pictures.labels.40.05_30")
    # label_path = os.path.expanduser("~/datasets/cacher/pictures.labels.40/pictures.labels.40.38_39")
    # 0.6合并的(现在都清理了白鹭郡测试数据)
    # label_path = os.path.expanduser("~/datasets/cacher/pictures.labels.35/pictures.labels.35.33_34")
    # 0.5合并的(现在都清理了白鹭郡测试数据)
    # label_path = os.path.expanduser("~/datasets/cacher/pictures.labels.48/pictures.labels.48.46_47")
    # 0.5合并的，并再次处理了剩余的(现在都清理了白鹭郡测试数据)
    # label_path = os.path.expanduser("~/datasets/cacher/pictures.labels.48/left_pictures.labels.48.46_47")
    # 0.6合并的,并且合并了剩余的，并踢出了0.5merge的(现在都清理了白鹭郡等所有测试数据)

    # label_path = os.path.expanduser("~/datasets/cacher/pictures.labels.35/left_pictures.labels.35.33_34.processed.v16")
    label_path = os.path.expanduser("/opt/cacher/faces_webface_112x112.labels")
    parser.add_argument('--label_path', default=label_path, help='training set directory')
    target = os.path.expanduser("~/datasets/maysa/lfw.bin")
    parser.add_argument('--target', type=str, default=target, help='verification targets')

    parser.add_argument('--load_weight', type=int, default=0, help='重新加载feature')
    parser.add_argument('--lr', type=float, default=0.0001, help='start learning rate')
    parser.add_argument('--per_batch_size', type=int, default=64, help='batch size in each context')

    # parser.add_argument('--pretrained', default='../models/model-r100-ii-1-16/model,29', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='../models/model-r34-7-19/model,172000', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='../models/r100-iccv/model,1', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='../models_retina100_2019-10-18/model,486201', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='./train/models_2019-11-06-14:24:12/model,492590', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='./train/models_2019-12-05-21:08:10/model,70060', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='./train/models_2019-12-12-23:04:29/model,9', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='./train/v26_2019-12-18-21:18:18/model,4', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='./train/v26_2019-12-20-17:26:24/model,9', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='./train/v28_2019-12-25-10:26:16/model,2', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='./train/model-r34-amf/model,0', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='./train/noise_2020-01-03-17:39:14/model,9', help='pretrained model to load')
    parser.add_argument('--pretrained', default='./train/v28_2019-12-25-15:26:45/model,8', help='pretrained model to load')
    # parser.add_argument('--pretrained', default='', help='pretrained model to load')

    parser.add_argument('--network', default='r100', help='specify network')
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--loss_type', type=int, default=4, help='loss type 5的时候为cos(margin_a*θ+margin_m) - margin_b;cos(θ+0.3)-0.2 or cos(θ+0.5)')
    parser.add_argument('--max_steps', type=int, default=0, help='max training batches')
    # parser.add_argument('--network', default='r100', help='specify network')
    parser.add_argument('--image-size', default='112,112', help='specify input image height and width')
    parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
    parser.add_argument('--version-input', type=int, default=1, help='network input config 1代表第一次卷积7x7-2改为3x3-1')
    parser.add_argument('--version_output', type=str, default='E', help='network embedding output config e代表的是bn-drop-fc-bn结构')
    parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config 3代表的是arc论文中对残差网络单元的修改，增加了更多bn和prelu')
    parser.add_argument('--version-multiplier', type=float, default=1.0, help='filters multiplier')
    parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
    parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')
    parser.add_argument('--lr_steps', type=str, default='', help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
    parser.add_argument('--fc7-lr-mult', type=float, default=1.0, help='lr mult for fc7')
    parser.add_argument("--fc7-no-bias", default=False, action="store_true", help="fc7 no bias flag")
    parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--margin_m', type=float, default=0.5, help='margin for loss,')
    parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
    parser.add_argument('--margin-a', type=float, default=1.0, help='')
    parser.add_argument('--margin-b', type=float, default=0.2, help='')
    parser.add_argument('--easy_margin', type=int, default=0, help='')
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
    gt_label = all_label
    extra_loss = None
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
        s = args.margin_s
        m = args.margin_m
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
        t = mx.sym.degrees(mx.sym.arccos(cos_t))
        origin_t = t
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
        # noise_tolerant

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
    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label=gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)
    out_list.append(mx.symbol.BlockGrad(mx.symbol.softmax(origin_fc7)))
    out_list.append(mx.symbol.BlockGrad(origin_t))
    out_list.append(mx.symbol.BlockGrad(cos_t))
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
    out = mx.symbol.Group(out_list)
    return (out, arg_params, aux_params)


def train_net(args):
    branch_name = git.Repo("..").active_branch.name
    prefix = time.strftime("%Y-%m-%d-%H:%M:%S")
    file_path = "train/{}_{}".format(branch_name, prefix)
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

    args.pretrained = os.path.expanduser(args.pretrained)

    ctx = [mx.gpu()]
    logging.info('use gpu0')
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    logging.info('num_layers %s', args.num_layers)
    logging.info('branch name %s', branch_name)
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

    if args.loss_type == 1 and args.num_classes > 20000:
        args.beta_freeze = 5000
        args.gamma = 0.06

    # data_shape = (args.image_channel, image_size[0], image_size[1])
    # dataset = FaceDataset(args.leveldb_path, args.label_path, leveldb_feature_path=os.path.expanduser("~/datasets/cacher/features"), min_images=10, max_images=3000000, ignore_labels={0})

    data_shape = (args.image_channel, image_size[0], image_size[1])
    leveldb_path = os.path.expanduser("~/datasets/cacher/pictures")
    label_path = os.path.expanduser("~/datasets/cacher/pictures.labels.35/left_pictures.labels.35.33_34.processed.v16")
    dataset1 = FaceDataset(leveldb_path, label_path, leveldb_feature_path=os.path.expanduser("~/datasets/cacher/features"), min_images=10, max_images=5, ignore_labels={0})

    leveldb_path = os.path.expanduser("~/datasets/cacher/ms1m-retina")
    label_path = os.path.expanduser("~/datasets/cacher/ms1m-retina.labels")
    dataset2 = FaceDataset(leveldb_path, label_path, min_images=10, max_images=5, ignore_labels={0})

    dataset = ListDataset(dataset1, dataset2)
    logging.info("dataset %s/%s", dataset.label_len, dataset.data_len)

    logging.info("dataset %s/%s", dataset.label_len, dataset.data_len)
    train_dataiter = FaceImageIter(
        batch_size=args.batch_size,
        data_shape=data_shape,
        dataset=dataset,
        shuffle=True,
        gauss=False
    )
    args.num_classes = train_dataiter.num_class()
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
        if "fc7_weight" in arg_params:
            logging.info("fc7_weight norm %s", mx.nd.norm(arg_params['fc7_weight'], axis=1).mean())
        if args.load_weight:
            arg_params['fc7_weight'] = dataset.label_features()
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    # label_name = 'softmax_label'
    # label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context=ctx,
        symbol=sym,
    )
    val_dataiter = None

    eval_metrics = [mx.metric.create([AccMetric(), AccMetric(True), LossMetric(), LossMetric(True), ThetaMetric()])]
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
    _rescale = 1.0 / args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    # opt.set_lr_mult(dict({k:0 for k in arg_params}))
    som = 20
    _cb = mx.callback.Speedometer(args.batch_size, som, auto_reset=True)

    ver_list = []
    ver_name_list = []
    path = os.path.join(args.target)
    if os.path.exists(path):
        name = os.path.basename(path)
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        logging.info('ver %s', name)
    else:
        logging.info("path %s not existed")

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
        # if args.loss_type >= 1 and args.loss_type <= 7:
        #     lr_steps = [100000, 140000, 160000]
        lr_steps = [8, 12, 16]
        lr_steps = [3]
    else:
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
    if len(lr_steps) == 1:
        end_epoch = 2 * lr_steps[-1]
    else:
        end_epoch = 2 * lr_steps[-1] - lr_steps[-2]
    epoch_sizes = [int(dataset.pic_len / args.batch_size)] * end_epoch
    args.max_steps = np.sum(epoch_sizes)
    args.lr_steps = lr_steps
    start_time = time.time()

    def save_png(cos_t_cur, end_str):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 8))
        ###绘图
        plt.hist(cos_t_cur, bins=1000, color='g')
        plt.title('余弦直方图')
        ###保存
        plt.savefig("{}/plt_{}.jpg".format(file_path, end_str))

    def save_mean_png(cos_t_cur, end_str):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 8))
        ###绘图
        x = np.arange(len(cos_t_cur))
        plt.plot(x, cos_t_cur, color="red", linewidth=1)
        plt.title('余弦柱状图')
        ###保存
        plt.savefig("{}/plt_mean_{}.jpg".format(file_path, end_str))

    def cal_thes(n2, end_str):
        sigm_l_th = int(len(n2) * 0.005)
        sigm_l = n2[sigm_l_th]
        sigm_r = n2[-sigm_l_th]
        logging.info("sigm_l %s sigm_r %s", sigm_l, sigm_r)

        bins = 100
        bin_dict = defaultdict(int)
        for n in n2:
            n = int(n * bins)
            if n == bins:
                n == bins - 1
            bin_dict[n] += 1

        n2 = np.zeros(bins * 2)
        for index in range(bins * 2):
            n2[index] = bin_dict[index - bins]
        n2_copy = n2.copy()
        for index in range(bins * 2):
            start = max(0, index - 2)
            end = min(bins * 2, index + 2)
            n2[index] = np.mean(n2_copy[start:end])

        save_mean_png(n2, end_str)
        max_thes = []
        max_counts = []
        for th in range(bins * 2):
            if 5 <= th <= bins * 2 - 6:
                is_max = True
                for i in range(11):
                    cur = n2[th - 5 + i]
                    if cur == 0 or cur > n2[th]:
                        is_max = False
                        break
                if is_max:
                    max_thes.append((th - bins) * 0.01)
                    max_counts.append(n2[th])
        logging.info("max_thes %s max_counts %s", max_thes, max_counts)

    def cal_noise(cos_t_cur, nbatch):
        save_png(cos_t_cur, nbatch)
        cos_t_cur.sort()
        cal_thes(cos_t_cur, nbatch)

    def cal_noise_end(cos_t_all, epoch):
        save_png(cos_t_all, "epoch_{}".format(epoch))
        cos_t_all.sort()
        cal_thes(cos_t_all, "epoch_{}".format(epoch))

    def _batch_callback(param):
        nbatch = param.nbatch
        global_batch = global_step[0]

        cos_t = model.get_outputs()[4].asnumpy()
        cos_ts.append(cos_t)
        # logging.info("cos_t %s ", cos_t)
        if global_batch % 1000 == 0:
            cos_t_cur = np.concatenate(cos_ts[-1000:])
            # bins = list(range(100))
            # bins = [b / 100 for b in bins]
            bins = 100
            sw.add_histogram(tag="cos_t", values=cos_t_cur, global_step=global_batch, bins=bins)
            cal_noise(cos_t_cur, global_batch)

        if nbatch != 0 and nbatch % 20 == 0:
            acc = param.eval_metric.get_name_value()[0][1]
            real_acc = param.eval_metric.get_name_value()[1][1]
            loss = param.eval_metric.get_name_value()[2][1]
            real_loss = param.eval_metric.get_name_value()[3][1]

            sw.add_scalar(tag='lr', value=opt.lr, global_step=global_batch)
            sw.add_scalar(tag='acc', value=acc, global_step=global_batch)
            sw.add_scalar(tag='real_acc', value=real_acc, global_step=global_batch)
            sw.add_scalar(tag='loss', value=loss, global_step=global_batch)
            sw.add_scalar(tag='real_loss', value=real_loss, global_step=global_batch)

            softmax = model.get_outputs()[1].asnumpy()
            real_softmax = model.get_outputs()[2].asnumpy()
            theta = model.get_outputs()[3].asnumpy()

            sw.add_histogram(tag="softmax_min", values=softmax.min(axis=1), global_step=global_batch, bins=100)
            sw.add_histogram(tag="softmax_max", values=softmax.max(axis=1), global_step=global_batch, bins=100)
            sw.add_histogram(tag="softmax", values=softmax, global_step=global_batch, bins=100)

            sw.add_histogram(tag="real_softmax_min", values=real_softmax.min(axis=1), global_step=global_batch, bins=100)
            sw.add_histogram(tag="real_softmax_max", values=real_softmax.max(axis=1), global_step=global_batch, bins=100)
            sw.add_histogram(tag="real_softmax", values=real_softmax, global_step=global_batch, bins=100)

            sw.add_histogram(tag="theta", values=theta, global_step=global_batch, bins=100)
            logging.info("softmax %s %s real_softmax %s %s", softmax.min(), softmax.max(), real_softmax.min(), real_softmax.max())
            # logging.info("base_lose %s extra_loss %s", base_lose, extra_loss)

            spend = (time.time() - start_time) / 3600
            if global_batch == 0:
                speed = 0
            else:
                speed = spend / global_batch
            left = (args.max_steps - global_step[0]) * speed
            logging.info('lr-batch-epoch: lr %s, nbatch/epoch_size %s/%s,  epoch %s, step %s spend/left %.02f/%.02f',
                         opt.lr, param.nbatch, int(dataset.pic_len / args.batch_size), param.epoch, global_step[0], spend, left)
            train_dataiter.print_info()
        # speed最后调用
        _cb(param)

        if global_batch <= args.beta_freeze:
            _beta = args.beta
        else:
            move = max(0, global_batch - args.beta_freeze)
            _beta = max(args.beta_min, args.beta * math.pow(1 + args.gamma * move, -1.0 * args.power))
        # logging.info('beta', _beta)
        os.environ['BETA'] = str(_beta)
        # global global_step
        global_step[0] += 1

    def epoch_cb(epoch, symbol, arg, aux):
        global cos_ts
        # 清理历史列表
        cos_t_all = np.concatenate(cos_ts)
        cal_noise_end(cos_t_all, epoch)
        cos_ts = cos_ts[-1000:]
        logging.info("================>epoch_cb epoch %s g_step %s args.lr_steps %s", epoch, global_step[0], args.lr_steps)
        _lr_steps = [step - 1 for step in args.lr_steps]
        for _lr in _lr_steps:
            if epoch == args.beta_freeze + _lr:
                opt.lr *= 0.1
                logging.info('lr change to %s', opt.lr)
                break
        if len(ver_list) > 0:
            acc_list = ver_test(epoch)
            logging.info('[%d]Accuracy-Highest: %s' % (epoch, acc_list))
            sw.add_scalar(tag='val', value=acc_list[0], global_step=global_step[0])

        logging.info('saving %s', epoch)
        arg, aux = model.get_params()
        new_arg = {}
        for k in arg:
            if k == "fc7_weight":
                continue
            new_arg[k] = arg[k]
        # 暂时不过滤weight
        new_arg = arg
        logging.info("fc7_weight norm %s", mx.nd.norm(new_arg['fc7_weight'], axis=1).mean())
        mx.model.save_checkpoint(prefix + "/model", epoch, model.symbol[0].get_children(), new_arg, aux)

        # 11-12w人
        # 10 100w
        # 50 310-350w
        # 100 500w
        # 改变学习率的第一个epoch不变
        #
        # if epoch == 0:
        #     dataset.max_images = 10
        #     dataset.reset()
        # if epoch == 1:
        #     dataset.max_images = 300
        #     dataset.reset()
        # # 下一个epoch才生效
        # for index in range(epoch + 1, end_epoch):
        #     epoch_sizes[index] = int(dataset.pic_len / args.batch_size)
        # args.max_steps = np.sum(epoch_sizes)
        # logging.info("================>change max_images to %s epoch %s g_step %s max_steps %s epoch_sizes %s ", dataset.max_images, epoch, global_step[0], epoch_sizes, args.max_steps)

    # train_dataiter = mx.io.PrefetchingIter(train_dataiter)
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


# 增加了对dataset的支持和对leveldb的支持
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
