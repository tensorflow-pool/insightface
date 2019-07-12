from __future__ import print_function

import logging
import os

import cv2
import mxnet as mx
import numpy as np

from utils.cost import Cost

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# MX_FB_MODEL_PATH = os.path.join(CURRENT_DIR, "../../model/faceboxes/mxnet/faceboxes_168_4_19.params")
MX_FB_MODEL_PATH = os.path.join(CURRENT_DIR, "mnet.25.full")

g_image_size = (1080, 1920)


def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs


class FaceDetector:
    def __init__(self, prefix=MX_FB_MODEL_PATH, epoch=0, img_size_w=1920, img_size_h=1080):
        self.logger = logging.getLogger("insight")
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        self.ctx = mx.gpu(0)
        self.arg_params, self.aux_params = ch_dev(arg_params, aux_params, self.ctx)
        self.net = sym

    def load_net(self):
        pass

    def resize(self, img_size_w, img_size_h):
        self.img_size = [img_size_h, img_size_w]

    def detect(self, img, threshold=0.5):
        cost = Cost("detect")
        im = img.astype(np.float32)
        data = im.transpose((0, 3, 1, 2))

        self.arg_params["data"] = data
        exe = self.net.bind(self.ctx, self.arg_params, args_grad=None, grad_req="null", aux_states=self.aux_params)
        exe.forward(is_train=False)
        dets_batch = exe.outputs[0]
        cost.record("forward")

        cost.end(show=False)
        return dets_batch

    # img is nv12
    def __call__(self, frames, score_threshold=0.5):
        cost = Cost("mx detect")

        h, w = self.img_size
        img = [np.frombuffer(hex_data, dtype=np.uint8) for hex_data in frames]
        cost.record("p1")
        batch = mx.nd.array(img, dtype=np.uint8)
        cost.record("p2")
        batch = batch.reshape((0, h, w, -1))
        origin_batch = batch.asnumpy()
        cost.record("p3")

        batch = batch.as_in_context(self.ctx)
        # batch = batch[:, :, :, [2, 1, 0]]
        cost.record("p4")

        det = self.detect(batch, score_threshold)
        # mx.nd.waitall()
        cost.record("detect")
        dets_numpy = det.asnumpy()
        cost.record("asnumpy")

        boxes_batch = []
        scores_batch = []
        landmarks_batch = []
        pos_batch = []
        for dets in dets_numpy:
            filter_indices = np.where(dets[:, 0] > score_threshold)[0]
            dets = dets[filter_indices]
            boxes = []
            scores = []
            landmarks = []
            poses = []
            for k in range(dets.shape[0]):
                xmin = int(dets[k, 1])
                ymin = int(dets[k, 2])
                xmax = int(dets[k, 3])
                ymax = int(dets[k, 4])
                score = dets[k, 0]
                if ymax - ymin < 10 or xmax - xmin < 10:
                    continue
                boxes.append([ymin, xmin, ymax, xmax])
                scores.append(score)
                landmarks.append(dets[k, 5:].reshape((5, 2)))
                pos, l, r, u, d = self.check_large_pose(dets[k, 5:].reshape((5, 2)), [xmin, ymin, xmax, ymax])
                eye_dis = np.linalg.norm(dets[k, 5:7] - dets[k, 7:9])
                sum_pos = eye_dis / (l + r + u + d)
                pos_val = sum_pos + 10000 if pos == 0 else sum_pos
                if eye_dis < 20:
                    pos_val = 0
                poses.append(pos_val)
            boxes_batch.append(np.array(boxes))
            scores_batch.append(np.array(scores))
            landmarks_batch.append(np.array(landmarks))
            pos_batch.append(np.array(poses))
        cost.record("filter")

        cost.end(show=False)
        return boxes_batch, scores_batch, landmarks_batch, pos_batch, origin_batch

    # img is nv12
    def detect_imgs(self, imgs, score_threshold=0.5):
        cost = Cost("mx detect")

        img = [np.frombuffer(hex_data, dtype=np.uint8) for hex_data in frames]
        cost.record("p1")
        batch = mx.nd.array(img, dtype=np.uint8)
        cost.record("p2")
        batch = batch.reshape((0, h, w, -1))
        origin_batch = batch.asnumpy()
        cost.record("p3")

        batch = batch.as_in_context(self.ctx)
        # batch = batch[:, :, :, [2, 1, 0]]
        cost.record("p4")

        det = self.detect(batch, score_threshold)
        # mx.nd.waitall()
        cost.record("detect")
        dets_numpy = det.asnumpy()
        cost.record("asnumpy")

        boxes_batch = []
        scores_batch = []
        landmarks_batch = []
        pos_batch = []
        for dets in dets_numpy:
            filter_indices = np.where(dets[:, 0] > score_threshold)[0]
            dets = dets[filter_indices]
            boxes = []
            scores = []
            landmarks = []
            poses = []
            for k in range(dets.shape[0]):
                xmin = int(dets[k, 1])
                ymin = int(dets[k, 2])
                xmax = int(dets[k, 3])
                ymax = int(dets[k, 4])
                score = dets[k, 0]
                if ymax - ymin < 10 or xmax - xmin < 10:
                    continue
                boxes.append([ymin, xmin, ymax, xmax])
                scores.append(score)
                landmarks.append(dets[k, 5:].reshape((5, 2)))
                pos, l, r, u, d = self.check_large_pose(dets[k, 5:].reshape((5, 2)), [xmin, ymin, xmax, ymax])
                eye_dis = np.linalg.norm(dets[k, 5:7] - dets[k, 7:9])
                sum_pos = eye_dis / (l + r + u + d)
                pos_val = sum_pos + 10000 if pos == 0 else sum_pos
                if eye_dis < 20:
                    pos_val = 0
                poses.append(pos_val)
            boxes_batch.append(np.array(boxes))
            scores_batch.append(np.array(scores))
            landmarks_batch.append(np.array(landmarks))
            pos_batch.append(np.array(poses))
        cost.record("filter")

        cost.end(show=False)
        return boxes_batch, scores_batch, landmarks_batch, pos_batch, origin_batch

    @staticmethod
    def check_large_pose(landmark, bbox):
        assert landmark.shape == (5, 2)
        assert len(bbox) == 4

        def get_theta(base, x, y):
            vx = x - base
            vy = y - base
            vx[1] *= -1
            vy[1] *= -1
            tx = np.arctan2(vx[1], vx[0])
            ty = np.arctan2(vy[1], vy[0])
            d = ty - tx
            d = np.degrees(d)
            # print(vx, tx, vy, ty, d)
            # if d<-1.*math.pi:
            #  d+=2*math.pi
            # elif d>math.pi:
            #  d-=2*math.pi
            if d < -180.0:
                d += 360.
            elif d > 180.0:
                d -= 360.0
            return d

        landmark = landmark.astype(np.float32)

        theta1 = get_theta(landmark[0], landmark[3], landmark[2])
        theta2 = get_theta(landmark[1], landmark[2], landmark[4])
        # print(va, vb, theta2)
        theta3 = get_theta(landmark[0], landmark[2], landmark[1])
        theta4 = get_theta(landmark[1], landmark[0], landmark[2])
        theta5 = get_theta(landmark[3], landmark[4], landmark[2])
        theta6 = get_theta(landmark[4], landmark[2], landmark[3])
        theta7 = get_theta(landmark[3], landmark[2], landmark[0])
        theta8 = get_theta(landmark[4], landmark[1], landmark[2])
        # print(theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8)
        left_score = 0.0
        right_score = 0.0
        up_score = 0.0
        down_score = 0.0
        if theta1 <= 0.0:
            left_score = 10.0
        elif theta2 <= 0.0:
            right_score = 10.0
        else:
            left_score = theta2 / theta1
            right_score = theta1 / theta2
        if theta3 <= 10.0 or theta4 <= 10.0:
            up_score = 10.0
        else:
            up_score = max(theta1 / theta3, theta2 / theta4)
        if theta5 <= 10.0 or theta6 <= 10.0:
            down_score = 10.0
        else:
            down_score = max(theta7 / theta5, theta8 / theta6)
        mleft = (landmark[0][0] + landmark[3][0]) / 2
        mright = (landmark[1][0] + landmark[4][0]) / 2
        box_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        ret = 0
        if left_score >= 3.0:
            ret = 1
        if ret == 0 and left_score >= 2.0:
            if mright <= box_center[0]:
                ret = 1
        if ret == 0 and right_score >= 3.0:
            ret = 2
        if ret == 0 and right_score >= 2.0:
            if mleft >= box_center[0]:
                ret = 2
        if ret == 0 and up_score >= 2.0:
            ret = 3
        if ret == 0 and down_score >= 5.0:
            ret = 4
        return ret, left_score, right_score, up_score, down_score


if __name__ == '__main__':
    cost = Cost("retina")
    face_detector = FaceDetector()
    face_detector.resize(1920, 1080)
    cost.record("init")

    path = "../../sample-images/t2.jpg"
    image_data = cv2.imread(path)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    strData = image_data.tobytes()

    boxes, score, img = face_detector.detect_imgs([image_data])
    cost.record("detect1")
    boxes, score, img = face_detector([strData, strData])
    cost.record("detect2")
    cost.end()
    # face_detector.save_model()
