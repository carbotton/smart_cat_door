from pathlib import Path

# =========================
# Hardware (GPIO)
# =========================
RELAY_PIN = 17  # BCM numbering

# =========================
# Door policy
# =========================
LOCK_DURATION_SECONDS = 15 * 60      # prey -> lock for 15 minutes
CLEAN_CONFIRMATIONS = 2              # no_prey must be confirmed twice (same event)

# If True, door will be forced open at startup until you toggle it off
OVERRIDE_DEFAULT_FORCE_OPEN = False

# =========================
# Vision inputs
# =========================
# Use an RTSP/IP camera URL, or set to 0 for the Pi camera via V4L2 (if available)
# Examples:
#   CAMERA_SOURCE = 0
#   CAMERA_SOURCE = "rtsp://user:pass@ip/stream"
CAMERA_SOURCE = 0  # laptop webcam

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
# Model assets
# =========================
# Point this at your *models/* directory.
MODELS_DIR = Path(__file__).resolve().parent / "models"

# TF Object Detection frozen graph (COCO SSD MobileNetV2).
# Find it with:
#   find ~ -name frozen_inference_graph.pb | grep ssdlite_mobilenet_v2_coco_2018_05_09
TFOD_FROZEN_GRAPH = Path.home() / "models/research/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"
