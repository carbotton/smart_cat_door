from pathlib import Path

# =========================
# Hardware (GPIO)
# =========================
RELAY_PIN = 17  # BCM numbering

# =========================
# Door policy
# =========================
LOCK_DURATION_SECONDS = 15 * 60      # prey -> lock for 15 minutes
CLEAN_CONFIRMATIONS = 1              # cummuli accumulator already confirms across frames

# If True, door will be forced open at startup until you toggle it off
OVERRIDE_DEFAULT_FORCE_OPEN = False

# =========================
# Debug
# =========================
SHOW_PREVIEW = False  # headless on Pi

# =========================
# Vision inputs
# =========================
# Use an RTSP/IP camera URL, or set to 0 for the Pi camera via V4L2 (if available)
# Examples:
#   CAMERA_SOURCE = 0
#   CAMERA_SOURCE = "rtsp://user:pass@ip/stream"
CAMERA_SOURCE = "rtsp://169.254.1.1:554/live/0/MAIN"

# If using RTSP, sometimes OpenCV needs a smaller buffer to reduce latency
CAP_PROP_BUFFERSIZE = 1

# Run cat detection / inference every N frames to reduce load
VISION_EVERY_N_FRAMES = 3

# If no cat is seen for this many consecutive inference cycles, we consider the "event" ended
EVENT_END_MISSES = 3

# Resolution
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360

# =========================
# Cummuli accumulator  (original Cat_Prey_Analyzer cascade.py)
# =========================
# Each face frame contributes:  50 - round(prey_conf * 100)
#   prey_conf=0.0  → +50 (strong no-prey evidence)
#   prey_conf=0.5  →   0 (neutral)
#   prey_conf=1.0  → -50 (strong prey evidence)
# Decision is made on the *average* contribution across face frames.
CUMULUS_NO_PREY_THRESHOLD =  2.9603   # avg > this → cat is clean → unlock
CUMULUS_PREY_THRESHOLD    = -10.0     # avg < this → prey confirmed → lock
CUMULUS_PATIENCE          =  2        # min face frames before any decision

# =========================
# Classifier thresholds
# =========================
# FaceFurClassifier: p <= FF_FACE_THRESHOLD is treated as "face confirmed".
# Lower = stricter gate (rejects more); raise to pass more borderline snout crops.
FF_FACE_THRESHOLD = 0.65   # default was hard-coded 0.50

# PreyClassifier: p > PREY_THRESHOLD is treated as "prey detected".
# Logs show correct prey frames clustering at 0.50-0.51, wrong ones at 0.34-0.49.
# Lowering to 0.40 captures those borderline-but-correct detections.
PREY_THRESHOLD = 0.40      # default was hard-coded 0.50

# =========================
# Model assets
# =========================
# Point this at your *models/* directory.
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Directory where decision snapshots are saved (created automatically).
SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"

# TF Object Detection frozen graph (COCO SSD MobileNetV2).
# Find it with:
#   find ~ -name frozen_inference_graph.pb | grep ssdlite_mobilenet_v2_coco_2018_05_09
TFOD_FROZEN_GRAPH = Path.home() / "models/research/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"
