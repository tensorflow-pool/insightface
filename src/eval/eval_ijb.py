import _pickle as cPickle
import glob
import os
import timeit
import warnings
from pathlib import Path

import cv2
import datetime
import os
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from skimage import transform as trans
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")

BaseDir = os.path.expanduser("~/datasets/ijb/IJB_release")


class Embedding2:
    def __init__(self, prefix, batch_size=64):
        print('loading', prefix)
        prefix, epoch = prefix.split(",")
        ctx = mx.gpu()
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        image_size = (112, 112)
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(for_training=False, data_shapes=[('data', (batch_size, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def get(self, data):
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        feat = self.model.get_outputs()[0]
        return feat


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


def batchify_fn(data):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    src[:, 0] += 8.0
    input_blob = []
    scores = []
    for item in data:
        rimg = item[0]
        landmark5 = item[1]
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg, M, (112, 112), borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        input_blob.append(img)
        scores.append(item[2])
    return mx.nd.array(input_blob, ctx=mx.gpu()), mx.nd.array(scores, mx.gpu()).astype(np.float32)
    #return mx.nd.array(input_blob, ctx=mx.context.Context('cpu_shared', 0)), mx.nd.array(scores, mx.context.Context('cpu_shared', 0)).astype(np.float32)


def get_image_feature_gluon(img_path, img_list_path, model_path, batch=32):
    embedding = Embedding2(model_path, batch)

    class ImageDataset(mx.gluon.data.Dataset):
        def __init__(self, img_path, img_list_path):
            super(ImageDataset, self).__init__()
            self.img_path = img_path
            self.img_list_path = img_list_path
            with open(img_list_path) as file:
                self.files = file.readlines()

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            each_line = self.files[idx]
            name_lmk_score = each_line.strip().split(' ')
            img_name = os.path.join(self.img_path, name_lmk_score[0])
            img = cv2.imread(img_name)
            lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            return img, lmk, name_lmk_score[-1]

    dataset = ImageDataset(img_path, img_list_path)
    print("dataset len ", len(dataset))
    # dataloader = mx.gluon.data.DataLoader(dataset, batch_size=batch, num_workers=2, thread_pool=True, batchify_fn=batchify_fn)
    dataloader = mx.gluon.data.DataLoader(dataset, batch_size=batch, num_workers=4, thread_pool=True, batchify_fn=batchify_fn)

    features = []
    faceness_scores = []
    for batch in print_progress(dataloader):
        features.append(embedding.get(batch[0]).asnumpy())
        faceness_scores.append(batch[1].asnumpy())
    print("1", datetime.datetime.now())
    ret_features = np.concatenate(features)
    print("2", datetime.datetime.now())
    ret_scores = np.concatenate(faceness_scores)
    print("3", datetime.datetime.now())
    return ret_features, ret_scores


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


def load_pair():
    # =============================================================
    # load template pairs for template-to-template verification
    # tid : template id,  label : 1/0
    # format:
    #           tid_1 tid_2 label
    # =============================================================
    start = timeit.default_timer()
    p1, p2, label = read_template_pair_list(os.path.join(BaseDir, 'IJBB/meta', 'ijbb_template_pair_label.txt'))
    stop = timeit.default_timer()
    print('ijbb_template_pair_label Time: %.2f s. ' % (stop - start))
    return p1, p2, label


def main1(model_path,score_save_name):
    start = timeit.default_timer()
    # img_feats = read_image_feature('./MS1MV2/IJBB_MS1MV2_r100_arcface.pkl')
    img_path = os.path.join(BaseDir, 'IJBB/loose_crop')
    img_list_path = os.path.join(BaseDir, 'IJBB/meta/ijbb_name_5pts_score.txt')
    img_feats, faceness_scores = get_image_feature_gluon(img_path, img_list_path, model_path)
    stop = timeit.default_timer()
    print('feature Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))
#     if not os.path.exists("ijb"):
#         os.mkdir("ijb")
#     np.save("ijb/features.npy", img_feats)
#     np.save("ijb/scores.npy", faceness_scores)

    # =============================================================
    # load image and template relationships for template feature embedding
    # tid --> template id,  mid --> media id
    # format:
    #           image_name tid mid
    # =============================================================
    start = timeit.default_timer()
    templates, medias = read_template_media_list(os.path.join(BaseDir, 'IJBB/meta', 'ijbb_face_tid_mid.txt'))
    stop = timeit.default_timer()
    print('ijbb_face_tid_mid Time: %.2f s. ' % (stop - start))

    p1, p2, label = load_pair()
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

    np.save(score_save_name, score)

    display(label, os.path.dirname(score_save_name))


def display(label, score_save_path=os.path.join(BaseDir, './IJBB/result3')):
    files = glob.glob(score_save_path + '/*.npy')
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
    print(tpr_fpr_table)
    #plt.show()
    fig.savefig(os.path.join(score_save_path, "plt.png"))

#os.path.join(BaseDir, 'pretrained_models/model-r100-ii-1-16/model, 29')
main1(model_path=os.path.join("../..", 'models/y2-iccv/model, 1'),
          score_save_name=os.path.join(BaseDir, 'IJBB/result3/ms1m_y2(N0D0F0).npy'))

# p1, p2, label = load_pair()
# display(label)
