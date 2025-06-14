# main.py

from door_controller import lock_door, unlock_door, door_cleanup
#from camera_handler import get_frame
#from detector import prey_detected
from logger import logger as log
#from override import let_in_flag

import time

let_in_flag = False
has_prey = False

def main():
    log.info("Smart Cat Door system started.")
    last_state = None

    try:
        while True:
            # TESTING: Check for real-time user input to toggle prey status
            user_input = input("Enter '1' to simulate prey detected, '0' to simulate no prey, or 'q' to quit: ").strip()

            if user_input == '1':
                has_prey = True
                log.info("Prey detected. Door will lock.")
            elif user_input == '0':
                has_prey = False
                log.info("No prey detected. Door will unlock.")
            elif user_input.lower() == 'q':
                log.info("Exiting program.")
                break
            else:
                log.info("Invalid input. Please enter '1', '0', or 'q'.")
                			
            # Check override switch first
            if let_in_flag:
                unlock_door()
                log.info("Override switch active. Door remains open.")
                time.sleep(1)
                continue

            # Capture a frame from the IP camera
            #frame = get_frame()
            #if frame is None:
            #    log.warning("No frame captured. Skipping iteration.")
            #    time.sleep(0.5)
            #    continue

            # Run prey detection on the frame
            #has_prey = prey_detected(frame)

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
        log.warning("Interrupted by user. Shutting down.")
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
    finally:
        door_cleanup()
        log.warning("System cleanup: Door unlocked.")

if __name__ == "__main__":
    main()
