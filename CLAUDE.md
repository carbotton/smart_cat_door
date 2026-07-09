# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smart Cat Door — a Raspberry Pi system that uses computer vision to detect prey carried by a cat and controls a solenoid door lock accordingly. The vision pipeline is fully self-contained in this repo.

## Running the system

```bash
python3 main.py
```

## Hardware

| Component | GPIO Pin | Notes |
|-----------|----------|-------|
| Solenoid relay | BCM 17 | `GPIO.LOW` = locked, `GPIO.HIGH` = unlocked |
| Override button | BCM 27 | Toggles `let_in_flag` via gpiozero interrupt |
| PIR motion sensor | BCM 26 | Used in test scripts only |
| IP camera | — | RTSP at `rtsp://169.254.1.1:554/live/0/MAIN` |

## Architecture

```
main.py              — entry point; spawns vision thread + timer thread
config.py            — all tunable parameters (pins, timing, camera URL, model paths)
door_controller.py   — GPIO interface: lock_door(), unlock_door(), door_cleanup()
override.py          — gpiozero button on BCM 27, toggles let_in_flag
logger.py            — writes to log/ (daily files: event_log_YYYYMMDD.log)

vision/
  cat_finder_tfod.py — TF-OD frozen graph (COCO SSD MobileNetV2) finds cat in frame
  pipeline.py        — VisionPipeline: given cat bounding box, runs face + prey analysis
  stages.py          — PreyClassifier, FaceFurClassifier, EyeDetector, HaarCatFace

models/
  Prey_Classifier/   — Keras VGG16 (.h5): is there prey in the snout?
  Face_Fur_Classifier/ — Keras MobileNet (.h5): confirms snout region is a face, not fur
  Eye_Detector/      — Keras model (.h5): locates eye midpoint to derive snout bounding box
  Haar_Classifier/   — OpenCV XML cascades for cat face detection
```

## Vision pipeline (per frame)

1. `CatFinderTFOD.detect()` — runs TF-OD frozen graph; returns bounding box if cat/dog (COCO class 17/18) found with score ≥ 0.45
2. `VisionPipeline.analyze()` — within the cat ROI:
   - Haar cascade tries to find cat face; falls back to `EyeDetector` snout estimation
   - `FaceFurClassifier` confirms the snout region is actually a face
   - `PreyClassifier` runs on the snout to return `prey: bool | None`
3. Result mapped to `"prey"` / `"no_prey"` / `"dk"` and passed to `door_decision_cb()`

## Door state machine (`main.py`)

- Default state: **locked**. Invariant: door is unlocked only during override or an active unlock window
- `"prey"` → lock for `LOCK_DURATION_SECONDS` (default 15 min); reset clean counter
- `"no_prey"` → increment clean counter; after `CLEAN_CONFIRMATIONS` (default 2) in same event, unlock for `UNLOCK_DURATION_SECONDS` (default 10 s), window extended while the cat is still seen clean; re-locks when it expires
- `"dk"` (don't know) → keep previous state
- Override button toggles force-open; override wins over all vision decisions; toggling it off re-locks on the next timer tick
- Timer thread (`timer_tick`) enforces both window expiries even when no new frames arrive
- Camera: after `CAMERA_MAX_READ_FAILS` consecutive failed reads, the capture is released and reopened (`CAMERA_RECONNECT_DELAY_SECONDS` between attempts); startup with camera absent retries the same way
- `test/test_door_state.py` — state machine checks, runs off-Pi (stubs RPi.GPIO)

## TF-OD frozen graph

**TF-OD** = TensorFlow Object Detection API. The frozen graph is the first stage of the vision pipeline — it answers "is there a cat in this frame, and where?"

- **Frozen graph (`.pb`)** — a TF1 format where model weights and architecture are baked into one binary file, inference-only. The code loads it via `tf.compat.v1.Session` (TF2's backward-compat layer).
- **COCO SSD MobileNetV2** — the specific model: trained on the 80-class COCO dataset (includes cats/dogs), SSD = Single Shot Detector architecture, MobileNetV2 = lightweight backbone suited for embedded hardware like the Pi.
- Only COCO classes 17 (cat) and 18 (dog) are acted on; everything else is ignored.

The frozen graph is **not bundled** in the repo (too large). Download it once:

```bash
mkdir -p /home/carbotton/models/research/object_detection
cd /home/carbotton/models/research/object_detection
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar -xzf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
```

The path is configured in `config.py`:
```python
TFOD_FROZEN_GRAPH = Path("/home/carbotton/models/research/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb")
```

All Keras `.h5` models and Haar XMLs are in `models/` and are bundled.

## Test scripts (`test/`)

| Script | Purpose |
|--------|---------|
| `test_solenoid.py` | Cycle relay ON/OFF via RPi.GPIO |
| `test_PIR_sensor.py` | Interrupt-driven PIR test via RPi.GPIO |
| `motion_record.py` | PIR-triggered RTSP recording (uses lgpio + OpenCV) |
| `record_camera.py` | Continuous RTSP recording in 15-min segments |
| `smart_cat_door.ipynb` | Google Colab — YOLOv8 training on cat dataset |
