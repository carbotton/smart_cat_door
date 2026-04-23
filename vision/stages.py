from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import tensorflow as tf
import tf_keras


class PreyClassifier:
    TARGET = 224

    def __init__(self, model_path: Path):
        def get_f1(y_true, y_pred):
            K = tf_keras.backend
            tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
            ppos = K.sum(K.round(K.clip(y_true, 0, 1)))
            precision = tp / (pp + K.epsilon())
            recall = tp / (ppos + K.epsilon())
            return 2 * (precision * recall) / (precision + recall + K.epsilon())

        custom = {"get_f1": get_f1} if "F1" in model_path.name else None
        self.model = tf_keras.models.load_model(str(model_path), custom_objects=custom)

    def predict(self, snout_bgr: np.ndarray) -> tuple[bool, float, float]:
        img = cv2.resize(snout_bgr, (self.TARGET, self.TARGET)) * (1.0 / 255.0)
        x = img.reshape((1, self.TARGET, self.TARGET, 3))
        t0 = time.time()
        p = float(self.model.predict(x, verbose=0)[0][0])
        dt = time.time() - t0
        return (p > 0.5), p, dt


class FaceFurClassifier:
    TARGET = 224

    def __init__(self, model_path: Path):
        self.model = tf_keras.models.load_model(str(model_path))

    def face_bool(self, snout_bgr: np.ndarray) -> tuple[bool, float, float]:
        img = cv2.resize(snout_bgr, (self.TARGET, self.TARGET)) * (1.0 / 255.0)
        x = img.reshape((1, self.TARGET, self.TARGET, 3))
        t0 = time.time()
        p = float(self.model.predict(x, verbose=0)[0][0])
        dt = time.time() - t0
        # Same behavior as fork: pred<=0.5 => True
        return (p <= 0.5), p, dt


class HaarCatFace:
    def __init__(self, xml_path: Path):
        self.cascade = cv2.CascadeClassifier(str(xml_path))

    def detect_local(self, roi_bgr: np.ndarray) -> tuple[np.ndarray, bool, float]:
        t0 = time.time()
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1, minSize=(25, 25))
        dt = time.time() - t0

        if len(faces) == 0:
            return np.array([[0, 0], [0, 0]], dtype=int), False, dt

        x, y, w, h = max(faces, key=lambda c: c[2] * c[3])
        xmin = int(x - w * 0.2); ymin = int(y - h * 0.4)
        xmax = int(x + w * 1.2); ymax = int(y + h * 1.6)
        return np.array([[xmin, ymin], [xmax, ymax]], dtype=int), True, dt


class EyeDetector:
    TARGET = 224

    def __init__(self, model_path: Path):
        self.model = tf_keras.models.load_model(str(model_path))

    def _letterbox(self, img: np.ndarray):
        old_h, old_w = img.shape[:2]
        ratio = float(self.TARGET) / max(old_h, old_w)
        new_h, new_w = int(old_h * ratio), int(old_w * ratio)
        resized = cv2.resize(img, (new_w, new_h))
        dw = self.TARGET - new_w
        dh = self.TARGET - new_h
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded, top, left, ratio

    def snout_box_full(self, frame_bgr: np.ndarray, cat_box: np.ndarray) -> tuple[np.ndarray, float]:
        (x1, y1), (x2, y2) = cat_box
        x1 = max(int(x1), 0); y1 = max(int(y1), 0)
        x2 = min(int(x2), frame_bgr.shape[1]); y2 = min(int(y2), frame_bgr.shape[0])

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([[0, 0], [0, 0]], dtype=int), 0.0

        pre, top, left, ratio = self._letterbox(roi)
        x = (pre.astype("float32") / 255.0).reshape((1, self.TARGET, self.TARGET, 3))

        t0 = time.time()
        pred = self.model.predict(x, verbose=0)[0].reshape((-1, 2))
        dt = time.time() - t0

        pred[:, 0] = (pred[:, 0] - left) / ratio + x1
        pred[:, 1] = (pred[:, 1] - top) / ratio + y1

        x_mid = int((pred[0, 0] + pred[1, 0]) / 2)
        y_mid = int((pred[0, 1] + pred[1, 1]) / 2)

        cc_w = abs(int(x2 - x1)); cc_h = abs(int(y2 - y1))
        cc_diff = max(cc_w, cc_h)

        top_margin = cc_diff / 4
        bottom_margin = cc_diff / 3
        side_margin = cc_diff / 4

        xmin = max(int(x_mid - side_margin), 0)
        ymin = max(int(y_mid - top_margin), 0)
        xmax = min(int(x_mid + side_margin), frame_bgr.shape[1])
        ymax = min(int(y_mid + bottom_margin), frame_bgr.shape[0])

        return np.array([[xmin, ymin], [xmax, ymax]], dtype=int), dt
