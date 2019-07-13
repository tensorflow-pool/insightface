# -*- coding: utf-8 -*-

import os
import numpy as np
import _pickle as cPickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import timeit
import sklearn
import cv2
import sys
import glob
sys.path.append(os.path.expanduser('~/datasets/ijb/IJB_release/recognition'))
from embedding import Embedding
from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Pathmake_grid
import warnings
warnings.filterwarnings("ignore")
import torchvision.utils as vutils

def get_image_feature(img_path, img_list_path, model_path, gpu_id):
    img_list = open(img_list_path)
    embedding = Embedding(model_path, 0, gpu_id)
    files = img_list.readlines()
    img_feats = []
    faceness_scores = []
    for img_index, each_line in enumerate(print_progress(files)):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape( (5,2) )
        img_feats.append(embedding.get(img,lmk))
        faceness_scores.append(name_lmk_score[-1])
        if img_index >= 10:
            break
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores

def read_template_media_list(path):
    ijb_meta = np.loadtxt(path, dtype=str)
    templates = ijb_meta[:,1].astype(np.int)
    medias = ijb_meta[:,2].astype(np.int)
    return templates, medias


def image2template_feature(img_feats = None, templates = None, medias = None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    print(img_feats.shape, templates.shape, medias.shape)
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        print("uqt ", uqt)
        (ind_t,) = np.where(templates == uqt)
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        print("face_medias ", face_medias)
        print("unique_medias ", unique_medias)
        print("unique_media_counts ", unique_media_counts)
        face_norm_feats = img_feats[ind_t]
        media_norm_feats = []
        for u,ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else: # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
    template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    return template_norm_feats, unique_templates

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
ppp =  os.path.expanduser(os.path.join('~/datasets/ijb/IJB_release/IJBB/meta', 'ijbb_face_tid_mid.txt'))
templates, medias = read_template_media_list(ppp)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))


# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
#img_feats = read_image_feature('./MS1MV2/IJBB_MS1MV2_r100_arcface.pkl')
img_path = os.path.expanduser('~/datasets/ijb/IJB_release/IJBB/loose_crop')
img_list_path = os.path.expanduser('~/datasets/ijb/IJB_release/IJBB/meta/ijbb_name_5pts_score.txt')
model_path = os.path.expanduser('~/datasets/ijb/IJB_release/pretrained_models/MS1MV2-ResNet100-Arcface/model')
gpu_id = 0
img_feats, faceness_scores = get_image_feature(img_path, img_list_path, model_path, gpu_id)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

use_norm_score = True  # if True, TestMode(N1)
use_detector_score = False  # if True, TestMode(D1)
use_flip_test = True  # if True, TestMode(F2)

if use_flip_test:
    # concat --- F1
    # img_input_feats = img_feats
    # add --- F2
    print(img_feats.shape)
    img_input_feats = img_feats[:, 0:int(img_feats.shape[1] / 2)] + img_feats[:, int(img_feats.shape[1] / 2):]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] / 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:, np.newaxis], 1, img_input_feats.shape[1])
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))