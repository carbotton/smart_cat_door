# main.py

import time
from door_controller import lock_door, unlock_door
from camera_handler import get_frame
from detector import prey_detected
from logger import get_logger
from override import let_in_flag

log = get_logger("main")

def main():
    log.info("Smart Cat Door system started.")
    last_state = None

    try:
        while True:
            # Check override switch first
            if let_in_flag:
                unlock_door()
                log.info("Override switch active. Door remains open.")
                time.sleep(1)
                continue

            # Capture a frame from the IP camera
            frame = get_frame()
            if frame is None:
                log.warning("No frame captured. Skipping iteration.")
                time.sleep(0.5)
                continue

            # Run prey detection on the frame
            has_prey = prey_detected(frame)

            # Only act on state change
            if has_prey != last_state:
                last_state = has_prey
                if has_prey:
                    lock_door()
                    log.info("Prey detected. Door locked.")
                else:
                    unlock_door()
                    log.info("Cat is clean. Door unlocked.")

            time.sleep(0.5)

    except KeyboardInterrupt:
        log.info("Interrupted by user. Shutting down.")
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
    finally:
        lock_door()
        log.info("System cleanup: Door locked for safety.")

if __name__ == "__main__":
    main()
