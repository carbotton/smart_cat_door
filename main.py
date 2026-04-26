import os
import threading
import time
from datetime import datetime
from typing import Optional, Union
import subprocess
import logging
import cv2

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "loglevel;quiet")

from logger import logger
from config import (
    LOCK_DURATION_SECONDS,
    CLEAN_CONFIRMATIONS,
    OVERRIDE_DEFAULT_FORCE_OPEN,
    STARTUP_UNLOCK_SECONDS,
    CAMERA_SOURCE,
    CAP_PROP_BUFFERSIZE,
    VISION_EVERY_N_FRAMES,
    EVENT_END_MISSES,
    MODELS_DIR,
    TFOD_FROZEN_GRAPH,
    SNAPSHOTS_DIR,
    CUMULUS_NO_PREY_THRESHOLD,
    CUMULUS_PREY_THRESHOLD,
    CUMULUS_PATIENCE,
)
from door_controller import lock_door, unlock_door, door_cleanup
import notifier

import override  # reads hardware button state (falls back gracefully when no GPIO)
override_btn = override.init_override_button(27)

from vision.cat_finder_tfod import CatFinderTFOD
from vision.pipeline import VisionPipeline


# ── Cummuli accumulator ─────────────────────────────────────────────────────

class CumulusAccumulator:
    """
    Per-event accumulator matching the original Cat_Prey_Analyzer cummuli system.

    Each face frame contributes:  50 - round(prey_conf * 100)
      conf=0.0 → +50 (strong no-prey),  conf=0.5 → 0,  conf=1.0 → -50 (strong prey)

    A decision is only made once face_count >= CUMULUS_PATIENCE:
      avg > NO_PREY_THRESHOLD  →  no_prey
      avg < PREY_THRESHOLD     →  prey
      otherwise                →  dk (still accumulating)

    After a prey/no_prey decision the caller should call reset() so the next
    cycle starts fresh within the same event.
    """

    def __init__(self):
        self.points: int = 0
        self.face_count: int = 0

    def reset(self):
        self.points = 0
        self.face_count = 0

    def update(self, prey_conf: float) -> int:
        """Record one face frame. Returns this frame's contribution (+/-)."""
        contribution = 50 - int(round(100 * prey_conf))
        self.points += contribution
        self.face_count += 1
        return contribution

    @property
    def avg(self) -> float:
        return self.points / self.face_count if self.face_count > 0 else 0.0

    def decide(self) -> str:
        """Returns 'prey', 'no_prey', or 'dk'."""
        if self.face_count < CUMULUS_PATIENCE:
            return "dk"
        a = self.avg
        if a > CUMULUS_NO_PREY_THRESHOLD:
            return "no_prey"
        if a < CUMULUS_PREY_THRESHOLD:
            return "prey"
        return "dk"

    def status_str(self) -> str:
        return f"cum avg={self.avg:+.2f} ({self.face_count} faces)"


# ── Door state machine ──────────────────────────────────────────────────────

_state_lock = threading.Lock()

door_locked: bool = True
lock_until_ts: Optional[float] = None
override_force_open: bool = OVERRIDE_DEFAULT_FORCE_OPEN

last_event_nr: Optional[int] = None
clean_hits_this_event: int = 0


def setup_ethernet_link_local(
    iface: str = "eth0",
    ip_addr: str = "169.254.1.2/16",
):
    logging.info("Setting up IP address for Ethernet...")
    try:
        subprocess.run(["sudo", "ip", "addr", "replace", ip_addr, "dev", iface], check=True)
        subprocess.run(["sudo", "ip", "link", "set", iface, "up"], check=True)
        logging.info(f"Ethernet {iface} configured with {ip_addr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to configure Ethernet {iface}: {e}")
        raise


def _apply_lock(reason: str):
    global door_locked
    lock_door()
    door_locked = True
    logger.info(f"DOOR CLOSED — {reason}")


def _apply_unlock(reason: str):
    global door_locked
    unlock_door()
    door_locked = False
    logger.info(f"DOOR OPENED — {reason}")


def door_decision_cb(decision: str, score=None, event_nr=None):
    global lock_until_ts, last_event_nr, clean_hits_this_event, override_force_open

    with _state_lock:
        override_force_open = bool(getattr(override, "let_in_flag", False)) or OVERRIDE_DEFAULT_FORCE_OPEN

        if override_force_open:
            _apply_unlock("override_force_open")
            return

        now = time.time()

        if lock_until_ts is not None and now < lock_until_ts:
            remaining = int(lock_until_ts - now)
            logger.info(f"  Prey lock active — {remaining}s remaining. Decision ignored.")
            return

        if lock_until_ts is not None and now >= lock_until_ts:
            lock_until_ts = None
            _apply_lock("prey_lock_expired")

        if event_nr is not None and event_nr != last_event_nr:
            last_event_nr = event_nr
            clean_hits_this_event = 0

        if decision == "prey":
            lock_until_ts = now + LOCK_DURATION_SECONDS
            clean_hits_this_event = 0
            _apply_lock(f"prey_detected — locked for {LOCK_DURATION_SECONDS}s")
            return

        if decision == "no_prey":
            clean_hits_this_event += 1
            logger.info(f"  Clean confirmation {clean_hits_this_event}/{CLEAN_CONFIRMATIONS}")
            if clean_hits_this_event >= CLEAN_CONFIRMATIONS:
                _apply_unlock("clean_confirmed")
                clean_hits_this_event = 0
            return

        logger.info("  DK — keeping previous door state.")


def timer_tick_forever(stop_event: threading.Event):
    global lock_until_ts, override_force_open

    while not stop_event.is_set():
        with _state_lock:
            override_force_open = bool(getattr(override, "let_in_flag", False)) or OVERRIDE_DEFAULT_FORCE_OPEN
            if override_force_open:
                if door_locked:
                    _apply_unlock("override_force_open_tick")
            else:
                now = time.time()
                if lock_until_ts is not None and now >= lock_until_ts:
                    lock_until_ts = None
                    _apply_lock("prey_lock_expired")
        time.sleep(0.5)


# ── Snapshot helper ─────────────────────────────────────────────────────────

def _save_snapshot(frame, label: str):
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOTS_DIR / f"{ts}_{label}.jpg"
    cv2.imwrite(str(path), frame)
    logger.info(f"  Snapshot saved: {path.name}")
    return path


# ── Camera helpers ──────────────────────────────────────────────────────────

def _open_capture(source: Union[int, str]):
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAP_PROP_BUFFERSIZE)
    except Exception:
        pass
    try:
        from config import CAPTURE_WIDTH, CAPTURE_HEIGHT
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    except Exception:
        pass
    return cap


# ── Vision loop ─────────────────────────────────────────────────────────────

def run_vision_forever(stop_event: threading.Event):
    cat_finder = CatFinderTFOD(TFOD_FROZEN_GRAPH)
    pipeline   = VisionPipeline(MODELS_DIR)

    cap = _open_capture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {CAMERA_SOURCE}")

    logger.info(f"Camera opened: {CAMERA_SOURCE}")
    logger.info(f"Cummuli thresholds — no_prey: >{CUMULUS_NO_PREY_THRESHOLD:.2f}  prey: <{CUMULUS_PREY_THRESHOLD:.1f}  patience: {CUMULUS_PATIENCE} faces")

    frame_idx        = 0
    event_nr         = 0
    miss_streak      = 0
    in_event         = False
    cum              = CumulusAccumulator()
    read_fail_streak = 0

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            read_fail_streak += 1
            if read_fail_streak % 10 == 1:
                logger.warning(f"Camera read failed (streak {read_fail_streak}); reconnecting...")
                cap.release()
                time.sleep(3)
                cap = _open_capture(CAMERA_SOURCE)
            else:
                time.sleep(0.1)
            continue
        read_fail_streak = 0

        frame_idx += 1

        if frame_idx % VISION_EVERY_N_FRAMES != 0:
            continue

        det = cat_finder.detect(frame)

        if not det.found or det.box is None:
            miss_streak += 1
            if miss_streak == 1 and in_event:
                logger.info(f"  Frame {frame_idx}: cat lost (miss streak starting)")
            elif miss_streak % 30 == 0:
                logger.info(f"  Frame {frame_idx}: no cat (miss streak {miss_streak})")
            if in_event and miss_streak >= EVENT_END_MISSES:
                logger.info(f"  Event #{event_nr} ended — {cum.status_str()} — no threshold reached")
                cum.reset()
                in_event = False
            if in_event:
                door_decision_cb("dk", score=None, event_nr=event_nr)
            continue

        # Cat found
        (x1, y1), (x2, y2) = det.box
        miss_streak = 0

        if not in_event:
            event_nr += 1
            in_event  = True
            cum.reset()
            logger.info(f">>> EVENT #{event_nr} START — cat detected")

        res = pipeline.analyze(frame, det.box, det.score)

        prey_str  = {True: "PREY", False: "no_prey", None: "dk"}.get(res.prey, "?")
        conf_str  = f" conf={res.prey_conf:.2f}" if res.prey_conf is not None else ""
        face_str  = f"face={res.face_method or 'no'}"
        ff_str    = (f" ff={'OK' if res.ff_confirmed else 'REJECTED'} ({res.ff_score:.2f})"
                     if res.ff_score is not None else " ff=n/a")
        infer_str = f" tfod={det.inference_s*1000:.0f}ms"

        if res.ff_confirmed and res.prey_conf is not None:
            contrib  = cum.update(res.prey_conf)
            cum_str  = f" | cum={contrib:+d} avg={cum.avg:+.2f} ({cum.face_count}f)"
        else:
            cum_str  = f" | cum=-- avg={cum.avg:+.2f} ({cum.face_count}f)" if cum.face_count else " | cum=--"

        logger.info(
            f"  Frame {frame_idx:5d} | cat={det.score:.2f} [{x1},{y1},{x2},{y2}]"
            f" | {face_str}{ff_str} | pred={prey_str}{conf_str}{infer_str}{cum_str}"
        )

        decision = cum.decide()
        if decision == "no_prey":
            logger.info(f"  >>> CUMULUS → NO PREY  ({cum.status_str()}) — unlocking")
            snap = _save_snapshot(frame, "no_prey")
            notifier.send_snapshot(snap, "no_prey")
            door_decision_cb("no_prey", score=cum.avg, event_nr=event_nr)
            cum.reset()
        elif decision == "prey":
            logger.info(f"  >>> CUMULUS → PREY  ({cum.status_str()}) — locking")
            snap = _save_snapshot(frame, "prey")
            notifier.send_snapshot(snap, "prey")
            door_decision_cb("prey", score=cum.avg, event_nr=event_nr)
            cum.reset()
        else:
            door_decision_cb("dk", score=cum.avg if cum.face_count else None, event_nr=event_nr)

    cap.release()


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    logger.info("Smart Cat Door system started.")

    setup_ethernet_link_local()

    if STARTUP_UNLOCK_SECONDS > 0:
        with _state_lock:
            _apply_unlock(f"startup_grace_{STARTUP_UNLOCK_SECONDS}s")
        logger.info(f"Startup grace period: door open for {STARTUP_UNLOCK_SECONDS}s.")
        time.sleep(STARTUP_UNLOCK_SECONDS)
        logger.info("Startup grace period ended — switching to vision control.")

    with _state_lock:
        _apply_lock("startup_default_locked")

    stop_event = threading.Event()

    vision_thread = threading.Thread(target=run_vision_forever, args=(stop_event,), daemon=True)
    vision_thread.start()
    logger.info("Vision thread started.")

    timer_thread = threading.Thread(target=timer_tick_forever, args=(stop_event,), daemon=True)
    timer_thread.start()
    logger.info("Timer thread started.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Shutting down.")
    finally:
        stop_event.set()
        door_cleanup()
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    main()
