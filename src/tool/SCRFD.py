import cv2 as cv
import numpy as np
from .deviceUtils import get_onnx_session


class SCRFD:
    """SCRFD (InsightFace det_10g) face detector; returns rows [x, y, w, h, 5 landmarks (10 values), score]."""

    def __init__(self, model_path, conf_threshold=0.4, nms_threshold=0.4, input_size=640):
        self.session = get_onnx_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf_threshold
        self.nms = nms_threshold
        self.size = input_size
        self.strides = [8, 16, 32]
        self.num_anchors = 2

    def detect_array(self, image, max_size=None):
        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:nh, :nw] = cv.resize(image, (nw, nh))
        blob = cv.dnn.blobFromImage(canvas, 1 / 128.0, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})
        n = len(self.strides)
        boxes, scores, kpss = [], [], []
        for i, stride in enumerate(self.strides):
            s = outs[i].reshape(-1)
            b = outs[i + n].reshape(-1, 4) * stride
            k = outs[i + n * 2].reshape(-1, 10) * stride
            g = self.size // stride
            ys, xs = np.mgrid[:g, :g]
            centers = np.stack([xs, ys], -1).reshape(-1, 2) * stride
            centers = np.repeat(centers, self.num_anchors, axis=0)
            keep = s >= self.conf
            c, s, b, k = centers[keep], s[keep], b[keep], k[keep]
            x1y1, x2y2 = c - b[:, :2], c + b[:, 2:]
            boxes.append(np.hstack([x1y1, x2y2]))
            scores.append(s)
            kpss.append(k.reshape(-1, 5, 2) + c[:, None, :])
        if not scores or sum(len(s) for s in scores) == 0:
            return np.zeros((0, 15))
        boxes, scores, kpss = np.vstack(boxes) / scale, np.concatenate(scores), np.vstack(kpss) / scale
        idx = cv.dnn.NMSBoxes([[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes],
                              scores.tolist(), self.conf, self.nms)
        idx = np.array(idx).reshape(-1)
        rows = [np.concatenate([[boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]],
                                kpss[i].reshape(-1), [scores[i]]]) for i in idx]
        return np.array(rows) if rows else np.zeros((0, 15))
