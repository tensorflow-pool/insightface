import _pickle as cPickle
import glob
import os
import sys
import timeit

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

sys.path.append('./recognition')
from embedding import Embedding2
from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BaseDir = os.path.expanduser("~/datasets/ijb/IJB_release")


def read_template_media_list(path):
    ijb_meta = np.loadtxt(path, dtype=str)
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    pairs = np.loadtxt(path, dtype=str)
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = cPickle.load(fid)
    return img_feats


def get_image_feature(img_path, img_list_path, model_path, gpu_id):
    img_list = open(img_list_path)
    batch = 64
    embedding = Embedding2(model_path, 0, gpu_id, batch)
    files = img_list.readlines()[:10]
    img_feats = []
    faceness_scores = []

    count = 0
    batchs = []
    for img_index, each_line in enumerate(print_progress(files)):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape((5, 2))

        batchs.append((img, lmk))
        faceness_scores.append(name_lmk_score[-1])
        if len(batchs) == batch:
            imgs = [k[0] for k in batchs]
            lmks = [k[1] for k in batchs]
            batchs = []
            img_feats.append(embedding.get(np.array(imgs), np.array(lmks)))
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
    template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), 1))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    print(total_sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = cPickle.load(fid)
    return img_feats


def main1():
    # =============================================================
    # load image and template relationships for template feature embedding
    # tid --> template id,  mid --> media id
    # format:
    #           image_name tid mid
    # =============================================================
    start = timeit.default_timer()
    templates, medias = read_template_media_list(os.path.join('IJBB/meta', 'ijbb_face_tid_mid.txt'))
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    # =============================================================
    # load template pairs for template-to-template verification
    # tid : template id,  label : 1/0
    # format:
    #           tid_1 tid_2 label
    # =============================================================
    start = timeit.default_timer()
    p1, p2, label = read_template_pair_list(os.path.join('IJBB/meta', 'ijbb_template_pair_label.txt'))
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    start = timeit.default_timer()
    # img_feats = read_image_feature('./MS1MV2/IJBB_MS1MV2_r100_arcface.pkl')
    img_path = './IJBB/loose_crop'
    img_list_path = './IJBB/meta/ijbb_name_5pts_score.txt'
    model_path = './pretrained_models/model-r34-amf/model'
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

    use_norm_score = False  # if True, TestMode(N1)
    use_detector_score = False  # if True, TestMode(D1)
    use_flip_test = False  # if True, TestMode(F2)

    if use_flip_test:
        # concat --- F1
        # img_input_feats = img_feats
        # add --- F2
        img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2] + img_feats[:, img_feats.shape[1] // 2:]
    else:
        img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

    if use_norm_score:
        img_input_feats = img_input_feats
    else:
        # normalise features to remove norm information
        img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))

    if use_detector_score:
        img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:, np.newaxis], 1, img_input_feats.shape[1])
    else:
        img_input_feats = img_input_feats

    print(img_input_feats.shape, templates.shape, medias.shape)
    template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    start = timeit.default_timer()
    score = verification(template_norm_feats, unique_templates, p1, p2)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    score_save_name = './IJBB/result2/MS1MV2-ResNet34-ArcFace-TestMode(N0D0F0).npy'
    np.save(score_save_name, score)

    score_save_path = './IJBB/result2'
    files = glob.glob(score_save_path + '/MS1MV2*.npy')
    methods = []
    scores = []
    for file in files:
        methods.append(Path(file).stem)
        scores.append(np.load(file))
    print(methods)
    methods = np.array(methods)
    scores = dict(zip(methods, scores))
    colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
    # x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    tpr_fpr_table = PrettyTable(['Methods'] + x_labels)
    fig = plt.figure()
    for method in methods:
        fpr, tpr, _ = roc_curve(label, scores[method])
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        plt.plot(fpr, tpr, color=colours[method], lw=1, label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc * 100)))
        tpr_fpr_row = []
        tpr_fpr_row.append(method)
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            tpr_fpr_row.append('%.4f' % tpr[min_index])
        tpr_fpr_table.add_row(tpr_fpr_row)
    plt.xlim([10 ** -6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on IJB-B')
    plt.legend(loc="lower right")
    plt.show()


def main2():
    start = timeit.default_timer()
    # img_feats = read_image_feature('./MS1MV2/IJBB_MS1MV2_r100_arcface.pkl')
    img_path = os.path.join(BaseDir, 'IJBB/loose_crop')
    img_list_path = os.path.join(BaseDir, 'IJBB/meta/ijbb_name_5pts_score.txt')
    model_path = os.path.join(BaseDir, 'pretrained_models/model-r34-amf/model')
    gpu_id = 0
    img_feats, faceness_scores = get_image_feature(img_path, img_list_path, model_path, gpu_id)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))


main2()
