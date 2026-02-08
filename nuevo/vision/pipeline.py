from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .stages import PreyClassifier, FaceFurClassifier, EyeDetector, HaarCatFace


@dataclass
class VisionResult:
    cat: bool
    cat_score: float | None
    cat_box: np.ndarray | None
    face: bool
    face_box: np.ndarray | None
    prey: bool | None
    prey_conf: float | None


class VisionPipeline:
    def __init__(self, models_dir: Path):
        self.pc = PreyClassifier(models_dir / "Prey_Classifier" / "F1_0.86_FINAL_VGG16_ownData_FTfrom15_350_Epochs_2020_06_14_12_11_25.h5")
        self.ff = FaceFurClassifier(models_dir / "Face_Fur_Classifier" / "256_05_mobileNet_50_Epochs_2020_05_07_14_56_25.h5")
        self.eye = EyeDetector(models_dir / "Eye_Detector" / "trainwhole100_Epochs_2020_04_30_18_05_25.h5")
        self.haar = HaarCatFace(models_dir / "Haar_Classifier" / "haarcascade_frontalcatface_extended.xml")

    def analyze(self, frame_bgr: np.ndarray, cat_box: np.ndarray, cat_score: float | None) -> VisionResult:
        (x1, y1), (x2, y2) = cat_box
        x1 = max(int(x1), 0); y1 = max(int(y1), 0)
        x2 = min(int(x2), frame_bgr.shape[1]); y2 = min(int(y2), frame_bgr.shape[0])
        roi = frame_bgr[y1:y2, x1:x2]

        face_box = None
        face = False

        # 1) Haar inside cat ROI
        if roi.size:
            bb_local, found, _ = self.haar.detect_local(roi)
            if found:
                bb = bb_local.copy()
                bb[:, 0] += x1
                bb[:, 1] += y1
                face_box = bb
                face = True

        # 2) Fallback: eye detector -> snout box
        if not face:
            bb, _ = self.eye.snout_box_full(frame_bgr, cat_box)
            (fx1, fy1), (fx2, fy2) = bb
            if fx2 > fx1 and fy2 > fy1:
                face_box = bb
                face = True

        prey = None
        prey_conf = None

        if face and face_box is not None:
            (fx1, fy1), (fx2, fy2) = face_box
            snout = frame_bgr[max(fy1,0):min(fy2,frame_bgr.shape[0]), max(fx1,0):min(fx2,frame_bgr.shape[1])]
            if snout.size:
                ff_ok, _, _ = self.ff.face_bool(snout)
                if ff_ok:
                    prey, prey_conf, _ = self.pc.predict(snout)

        return VisionResult(
            cat=True,
            cat_score=cat_score,
            cat_box=cat_box,
            face=face,
            face_box=face_box,
            prey=prey,
            prey_conf=prey_conf,
        )
