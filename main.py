import threading
import time
from typing import Optional, Union
import subprocess
import logging
import cv2

from logger import logger
from config import (
    LOCK_DURATION_SECONDS,
    CLEAN_CONFIRMATIONS,
    UNLOCK_DURATION_SECONDS,
    OVERRIDE_DEFAULT_FORCE_OPEN,
    CAMERA_SOURCE,
    CAP_PROP_BUFFERSIZE,
    CAMERA_MAX_READ_FAILS,
    CAMERA_RECONNECT_DELAY_SECONDS,
    VISION_EVERY_N_FRAMES,
    EVENT_END_MISSES,
    MODELS_DIR,
    TFOD_FROZEN_GRAPH,
)
from door_controller import lock_door, unlock_door, door_cleanup

import os
os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")


import override  # reads hardware button state
override_btn = override.init_override_button(27)

from vision.cat_finder_tfod import CatFinderTFOD
from vision.pipeline import VisionPipeline


# --- Door state machine (MVP) ---
_state_lock = threading.Lock()

door_locked: bool = True
lock_until_ts: Optional[float] = None    # prey lock window
unlock_until_ts: Optional[float] = None  # clean unlock window
override_force_open: bool = OVERRIDE_DEFAULT_FORCE_OPEN

last_event_nr: Optional[int] = None
clean_hits_this_event: int = 0


def setup_ethernet_link_local(
    iface: str = "eth0",
    ip_addr: str = "169.254.1.2/16",
):
    logging.info("Setting up IP address for Ethernet...")
    try:
        subprocess.run(
            ["sudo", "ip", "addr", "replace", ip_addr, "dev", iface],
            check=True,
        )
        subprocess.run(
            ["sudo", "ip", "link", "set", iface, "up"],
            check=True,
        )
        logging.info(f"Ethernet {iface} configured with {ip_addr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to configure Ethernet {iface}: {e}")
        raise


def _apply_lock(reason: str):
    global door_locked
    lock_door()
    door_locked = True
    logger.info(f"Door -> LOCKED ({reason})")


def _apply_unlock(reason: str):
    global door_locked
    unlock_door()
    door_locked = False
    logger.info(f"Door -> UNLOCKED ({reason})")


def door_decision_cb(decision: str, score=None, event_nr=None):
    """
    decision: "prey" | "no_prey" | "dk"
    """
    global lock_until_ts, unlock_until_ts, last_event_nr, clean_hits_this_event, override_force_open

    with _state_lock:
        logger.info(f"Vision decision: {decision} score={score} event={event_nr}")

        # Sync override from button
        override_force_open = bool(getattr(override, "let_in_flag", False)) or OVERRIDE_DEFAULT_FORCE_OPEN

        if override_force_open:
            _apply_unlock("override_force_open")
            return

        now = time.time()

        # If we are in a prey lock window, ignore everything until it expires
        if lock_until_ts is not None and now < lock_until_ts:
            logger.info("Ignoring decision (prey lock window active).")
            return

        # If lock window expired, return to default LOCKED
        if lock_until_ts is not None and now >= lock_until_ts:
            lock_until_ts = None
            _apply_lock("prey_lock_expired_default_locked")

        # Track event changes for "2x clean in same event"
        if event_nr is not None and event_nr != last_event_nr:
            last_event_nr = event_nr
            clean_hits_this_event = 0

        if decision == "prey":
            lock_until_ts = now + LOCK_DURATION_SECONDS
            unlock_until_ts = None
            clean_hits_this_event = 0
            _apply_lock(f"prey_detected_lock_{LOCK_DURATION_SECONDS}s")
            return

        if decision == "no_prey":
            clean_hits_this_event += 1
            logger.info(f"Clean confirmations: {clean_hits_this_event}/{CLEAN_CONFIRMATIONS}")
            if clean_hits_this_event >= CLEAN_CONFIRMATIONS:
                # keep extending the window while the cat is still seen clean
                unlock_until_ts = now + UNLOCK_DURATION_SECONDS
                if door_locked:
                    _apply_unlock("clean_confirmed")
            return

        # dk => keep previous state
        logger.info("DK decision: keeping previous door state.")


def timer_tick(now: Optional[float] = None):
    """Enforces window expiry even if no new vision decisions arrive.

    Invariant: door is unlocked only during override or an active unlock window.
    """
    global lock_until_ts, unlock_until_ts, override_force_open

    with _state_lock:
        override_force_open = bool(getattr(override, "let_in_flag", False)) or OVERRIDE_DEFAULT_FORCE_OPEN

        if override_force_open:
            if door_locked:
                _apply_unlock("override_force_open_tick")
            return

        if now is None:
            now = time.time()

        if lock_until_ts is not None and now >= lock_until_ts:
            lock_until_ts = None
            _apply_lock("prey_lock_expired_default_locked")
        elif not door_locked and (unlock_until_ts is None or now >= unlock_until_ts):
            # covers both: unlock window expired, and override just turned off
            unlock_until_ts = None
            _apply_lock("unlock_expired_default_locked")


def timer_tick_forever(stop_event: threading.Event):
    while not stop_event.is_set():
        timer_tick()
        time.sleep(0.5)


def _open_capture(source: Union[int, str]):
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAP_PROP_BUFFERSIZE)
    except Exception:
        pass

    # NEW: request smaller frames
    try:
        from config import CAPTURE_WIDTH, CAPTURE_HEIGHT
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    except Exception:
        pass

    return cap


def run_vision_forever(stop_event: threading.Event):
    """
    Runs vision loop. No fork imports. Uses:
      - TF-OD cat/dog detector (frozen graph)
      - Haar + Eye + Face/Fur + Prey classifiers
    """
    cat_finder = CatFinderTFOD(TFOD_FROZEN_GRAPH)
    pipeline = VisionPipeline(MODELS_DIR)

    cap = _open_capture(CAMERA_SOURCE)
    if not cap.isOpened():
        logger.warning(f"Camera source not available at startup: {CAMERA_SOURCE}; will keep retrying.")

    frame_idx = 0
    event_nr = 0
    miss_streak = 0
    in_event = False
    read_fails = 0

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            read_fails += 1
            if read_fails >= CAMERA_MAX_READ_FAILS:
                logger.warning(f"Camera read failed {read_fails}x; reconnecting to {CAMERA_SOURCE}...")
                cap.release()
                time.sleep(CAMERA_RECONNECT_DELAY_SECONDS)
                cap = _open_capture(CAMERA_SOURCE)
                read_fails = 0
            else:
                time.sleep(0.1)
            continue
        read_fails = 0

        frame_idx += 1
        if frame_idx % VISION_EVERY_N_FRAMES != 0:
            continue

        det = cat_finder.detect(frame)

        if not det.found or det.box is None:
            miss_streak += 1
            if in_event and miss_streak >= EVENT_END_MISSES:
                in_event = False
            # No cat: only emit dk when an event is active (otherwise keep quiet)
            if in_event:
                door_decision_cb("dk", score=None, event_nr=event_nr)
            continue

        # Cat found
        miss_streak = 0
        if not in_event:
            event_nr += 1
            in_event = True

        res = pipeline.analyze(frame, det.box, det.score)

        if res.prey is True:
            door_decision_cb("prey", score=res.prey_conf, event_nr=event_nr)
        elif res.prey is False:
            door_decision_cb("no_prey", score=res.prey_conf, event_nr=event_nr)
        else:
            door_decision_cb("dk", score=None, event_nr=event_nr)

    cap.release()


def main():
    logger.info("Smart Cat Door system started.")
    
    setup_ethernet_link_local()

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
        logger.warning("System cleanup: GPIO cleaned up.")


if __name__ == "__main__":
    main()
