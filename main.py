import os
import threading
import time
import subprocess
import sys
from pathlib import Path
from typing import Optional

from logger import logger
from config import LOCK_DURATION_SECONDS, CLEAN_CONFIRMATIONS, OVERRIDE_DEFAULT_FORCE_OPEN
from door_controller import lock_door, unlock_door, door_cleanup

# --- Paths ---
CAT_PREY_DIR = Path("/home/carbotton/smart_cat_door/Cat_Prey_Analyzer")
STARTER_SH = CAT_PREY_DIR / "catCam_starter.sh"

# MVP: fail fast if wrong
if not (CAT_PREY_DIR / "cascade.py").exists():
    raise FileNotFoundError(f"Missing {CAT_PREY_DIR}/cascade.py")

# Critical: set cwd before importing cascade (your code relies on os.getcwd())
os.chdir(str(CAT_PREY_DIR))

# Make import work
sys.path.insert(0, str(CAT_PREY_DIR))

from cascade import Sequential_Cascade_Feeder


# --- Door state machine (MVP) ---
_state_lock = threading.Lock()

door_locked: bool = True
lock_until_ts: Optional[float] = None     # absolute lock window for prey
override_force_open: bool = OVERRIDE_DEFAULT_FORCE_OPEN

last_event_nr: Optional[int] = None
clean_hits_this_event: int = 0


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
    Called by Cat_Prey_Analyzer (in-process).
    decision: "prey" | "no_prey" | "dk"
    """
    global lock_until_ts, last_event_nr, clean_hits_this_event, override_force_open

    with _state_lock:
        logger.info(f"Vision decision: {decision} score={score} event={event_nr}")

        # Override wins (for now, force open if enabled)
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
            clean_hits_this_event = 0
            _apply_lock(f"prey_detected_lock_{LOCK_DURATION_SECONDS}s")
            return

        if decision == "no_prey":
            # require 2 confirmations in the same event
            clean_hits_this_event += 1
            logger.info(f"Clean confirmations: {clean_hits_this_event}/{CLEAN_CONFIRMATIONS}")

            if clean_hits_this_event >= CLEAN_CONFIRMATIONS:
                _apply_unlock("clean_confirmed")
            return

        if decision == "dk":
            # keep previous state
            logger.info("DK decision: keeping previous door state.")
            return


def start_ip_setup():
    # run once at startup
    subprocess.run(["/bin/bash", str(STARTER_SH)], check=True)
    logger.info("IP setup completed.")


def run_vision_forever(stop_event: threading.Event):
    """
    Runs vision; if it crashes, logs and restarts it.
    """
    # Cat_Prey_Analyzer expects cwd to be its folder (relative paths/logs/etc.)
    os.chdir(str(CAT_PREY_DIR))

    while not stop_event.is_set():
        try:
            sq = Sequential_Cascade_Feeder(door_decision_cb=door_decision_cb)
            sq.queque_handler()
        except Exception as e:
            logger.exception(f"Vision crashed: {e}. Restarting in 2 seconds...")
            time.sleep(2)


def timer_tick_forever(stop_event: threading.Event):
    """
    Enforces prey lock expiry even if no new vision decisions arrive.
    """
    global lock_until_ts, override_force_open

    while not stop_event.is_set():
        with _state_lock:
            if override_force_open:
                if door_locked:
                    _apply_unlock("override_force_open_tick")
            else:
                now = time.time()
                if lock_until_ts is not None and now >= lock_until_ts:
                    lock_until_ts = None
                    _apply_lock("prey_lock_expired_default_locked")
        time.sleep(0.5)


def main():
    logger.info("Smart Cat Door system started.")

    # Default state: locked
    with _state_lock:
        _apply_lock("startup_default_locked")

    # 1) IP setup once at startup
    try:
        start_ip_setup()
    except Exception as e:
        logger.exception(f"IP setup failed: {e}")
        # continue anyway (lets you debug without camera)

    # 2) Start vision with auto-restart
    stop_event = threading.Event()

    vision_thread = threading.Thread(target=run_vision_forever, args=(stop_event,), daemon=True)
    vision_thread.start()
    logger.info("Vision thread started (auto-restart enabled).")

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
