# smart_cat_door (MVP)

This project controls a cat door lock based on camera vision:
- If prey is detected -> door locks for a configurable time window
- If cat is clean -> door unlocks (after N confirmations)
- Manual override button can force the door open

## Key design point
This codebase **does not depend on the forked Cat_Prey_Analyzer logic**.
You only reuse the **model assets** (the `.h5` files + Haar XML), and a TF Object Detection frozen graph for cat detection.

## Folder layout
- `main.py`: system entrypoint (vision loop + door policy)
- `door_controller.py`: GPIO output control
- `override.py`: GPIO27 button toggles force-open
- `vision/`: all vision logic (clean and owned)
- `models/`: model assets directory (copy from your Cat_Prey_Analyzer/models)

## Setup
### 1) Copy model assets
Copy your existing models folder into this project:

```bash
cp -a /home/carbotton/smart_cat_door/Cat_Prey_Analyzer/models ./models
```

### 2) Set TF-OD frozen graph path
Edit `config.py` and set `TFOD_FROZEN_GRAPH` to the absolute path of:

`ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb`

Find it with:
```bash
find ~ -name frozen_inference_graph.pb | grep ssdlite_mobilenet_v2_coco_2018_05_09
```

### 3) Run
```bash
python3 main.py
```

## Notes
- `CAMERA_SOURCE` can be `0` (local camera) or an RTSP URL string.
- If you want lower latency on RTSP, keep `CAP_PROP_BUFFERSIZE=1`.
