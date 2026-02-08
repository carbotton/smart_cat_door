from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
import tensorflow as tf


@dataclass
class CatDetection:
    found: bool
    box: np.ndarray | None   # [[xmin,ymin],[xmax,ymax]] int
    score: float | None
    inference_s: float


class CatFinderTFOD:
    """
    TensorFlow Object Detection (frozen graph) cat/dog finder.
    Looks for COCO class ids: cat=17, dog=18.
    """

    def __init__(self, frozen_graph_pb: Path):
        if not frozen_graph_pb.exists():
            raise FileNotFoundError(f"Frozen graph not found: {frozen_graph_pb}")

        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(str(frozen_graph_pb), "rb") as f:
                od_graph_def.ParseFromString(f.read())
            tf.import_graph_def(od_graph_def, name="")

        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.image_tensor = self.graph.get_tensor_by_name("image_tensor:0")
        self.boxes = self.graph.get_tensor_by_name("detection_boxes:0")
        self.scores = self.graph.get_tensor_by_name("detection_scores:0")
        self.classes = self.graph.get_tensor_by_name("detection_classes:0")
        self.num = self.graph.get_tensor_by_name("num_detections:0")

    def detect(self, frame_bgr: np.ndarray, score_thresh: float = 0.45, max_checks: int = 10) -> CatDetection:
        # TF-OD expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(rgb, axis=0)

        t0 = time.time()
        boxes, scores, classes, _ = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num],
            feed_dict={self.image_tensor: inp},
        )
        dt = time.time() - t0

        scores_ = scores[0]
        classes_ = classes[0].astype(np.int32)
        boxes_ = boxes[0]

        best_idx = None
        best_score = 0.0

        for i in range(min(max_checks, len(scores_))):
            if float(scores_[i]) < score_thresh:
                continue
            if classes_[i] in (17, 18):  # cat or dog
                if float(scores_[i]) > best_score:
                    best_score = float(scores_[i])
                    best_idx = i

        if best_idx is None:
            return CatDetection(found=False, box=None, score=None, inference_s=dt)

        ymin, xmin, ymax, xmax = boxes_[best_idx]
        h, w = frame_bgr.shape[:2]
        box = np.array([[int(xmin * w), int(ymin * h)], [int(xmax * w), int(ymax * h)]], dtype=int)
        return CatDetection(found=True, box=box, score=best_score, inference_s=dt)
