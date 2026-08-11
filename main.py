import os
import threading
import time
from datetime import datetime
from typing import Optional, Union
import subprocess
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

# While time.time() < grace_until_ts the door stays open and vision decisions
# are ignored (vision thread still runs so models warm up during the grace).
grace_until_ts: float = 0.0

last_event_nr: Optional[int] = None
clean_hits_this_event: int = 0


def setup_ethernet_link_local(
    iface: str = "eth0",
    ip_addr: str = "169.254.1.2/16",
):
    logger.info("Setting up IP address for Ethernet...")
    try:
        subprocess.run(["sudo", "ip", "addr", "replace", ip_addr, "dev", iface], check=True)
        subprocess.run(["sudo", "ip", "link", "set", iface, "up"], check=True)
        logger.info(f"Ethernet {iface} configured with {ip_addr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to configure Ethernet {iface}: {e}")
        raise


def _apply_lock(reason: str):
    global door_locked
    # lock_door()  # ponytail: view-mode, no door hardware attached
    door_locked = True
    logger.info(f"DOOR CLOSED — {reason}")


def _apply_unlock(reason: str):
    global door_locked
    # unlock_door()  # ponytail: view-mode, no door hardware attached
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

        if now < grace_until_ts:
            logger.info("  Startup grace active — decision ignored, door stays open.")
            return

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


def _snapshot_and_notify_async(frame, label: str):
    """Write snapshot + send Telegram off the vision thread (upload can block 15s)."""
    def _work(img):
        try:
            path = _save_snapshot(img, label)
            notifier.send_snapshot(path, label)
        except Exception as e:
            logger.warning(f"  Snapshot/notify failed: {e}")

    threading.Thread(target=_work, args=(frame.copy(),), daemon=True).start()


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


class _LatestFrameReader:
    """Background thread that drains the RTSP buffer, always keeping the newest frame.

    Frames carry a sequence number so consumers can block until a *new* frame
    arrives (no busy loop, no analyzing the same frame twice). If the camera
    dies the sequence stops advancing and wait_for_frame() times out, so a
    frozen last frame is never re-delivered.
    """

    RECONNECT_AFTER_FAILS = 10

    def __init__(self, source: Union[int, str]):
        self._source = source
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._cap = _open_capture(source)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fail_streak = 0
        while True:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
                fail_streak = 0
            else:
                fail_streak += 1
                if fail_streak >= self.RECONNECT_AFTER_FAILS:
                    logger.warning(f"Camera read failed {fail_streak}x; reconnecting...")
                    self._cap.release()
                    time.sleep(3)
                    self._cap = _open_capture(self._source)
                    fail_streak = 0
                else:
                    time.sleep(0.1)

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0):
        """Block until a frame newer than last_seq exists.

        Returns (frame_copy, seq) or (None, last_seq) on timeout.
        """
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout)
            if self._seq <= last_seq or self._frame is None:
                return None, last_seq
            return self._frame.copy(), self._seq


# ── Vision loop ─────────────────────────────────────────────────────────────

def run_vision_forever(stop_event: threading.Event):
    cat_finder = CatFinderTFOD(TFOD_FROZEN_GRAPH)
    pipeline   = VisionPipeline(MODELS_DIR)

    reader = _LatestFrameReader(CAMERA_SOURCE)

    # Wait for the first frame before declaring the camera open. Retries
    # forever (camera may power up slower than the Pi after a shared outage).
    first_seq = 0
    waited = 0
    while True:
        frame, first_seq = reader.wait_for_frame(first_seq, timeout=1.0)
        if frame is not None:
            break
        waited += 1
        if waited % 30 == 0:
            logger.warning(f"Still waiting for camera source: {CAMERA_SOURCE} ({waited}s)")

    logger.info(f"Camera opened: {CAMERA_SOURCE}")
    logger.info(f"Cummuli thresholds — no_prey: >{CUMULUS_NO_PREY_THRESHOLD:.2f}  prey: <{CUMULUS_PREY_THRESHOLD:.1f}  patience: {CUMULUS_PATIENCE} faces")

    frame_idx   = 0
    last_seq    = first_seq
    event_nr    = 0
    miss_streak = 0
    in_event    = False
    cum         = CumulusAccumulator()

    while not stop_event.is_set():
        frame, last_seq = reader.wait_for_frame(last_seq, timeout=1.0)
        if frame is None:
            continue

        frame_idx += 1

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
            _snapshot_and_notify_async(frame, "test")  # TEMP: camera-angle test, remove once camera is repositioned

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
            door_decision_cb("no_prey", score=cum.avg, event_nr=event_nr)
            _snapshot_and_notify_async(frame, "no_prey")
            cum.reset()
        elif decision == "prey":
            logger.info(f"  >>> CUMULUS → PREY  ({cum.status_str()}) — locking")
            door_decision_cb("prey", score=cum.avg, event_nr=event_nr)
            _snapshot_and_notify_async(frame, "prey")
            cum.reset()
        else:
            door_decision_cb("dk", score=cum.avg if cum.face_count else None, event_nr=event_nr)


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    global grace_until_ts

    logger.info("Smart Cat Door system started.")

    setup_ethernet_link_local()

    if STARTUP_UNLOCK_SECONDS > 0:
        grace_until_ts = time.time() + STARTUP_UNLOCK_SECONDS
        with _state_lock:
            _apply_unlock(f"startup_grace_{STARTUP_UNLOCK_SECONDS}s")
        logger.info(f"Startup grace period: door open for {STARTUP_UNLOCK_SECONDS}s.")

    stop_event = threading.Event()

    # Start vision immediately so model loading (slow on the Pi) overlaps the
    # grace period; door_decision_cb ignores decisions until the grace ends.
    vision_thread = threading.Thread(target=run_vision_forever, args=(stop_event,), daemon=True)
    vision_thread.start()
    logger.info("Vision thread started.")

    timer_thread = threading.Thread(target=timer_tick_forever, args=(stop_event,), daemon=True)
    timer_thread.start()
    logger.info("Timer thread started.")

    if STARTUP_UNLOCK_SECONDS > 0:
        time.sleep(max(0.0, grace_until_ts - time.time()))
        logger.info("Startup grace period ended — switching to vision control.")

    with _state_lock:
        if not (bool(getattr(override, "let_in_flag", False)) or OVERRIDE_DEFAULT_FORCE_OPEN):
            _apply_lock("startup_default_locked")

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
