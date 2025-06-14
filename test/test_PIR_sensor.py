import RPi.GPIO as GPIO
import time
import signal
import sys

PIR_PIN = 26  # Your PIR sensor pin

# Cleanup GPIO before setting up
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


def motion_detected(channel):
    print("?? Motion detected!")

# Function to cleanup GPIO when exiting
def cleanup_gpio(signal, frame):
    print("\nCleaning up GPIO and exiting...")
    GPIO.cleanup()
    sys.exit(0)

# Register cleanup function for Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, cleanup_gpio)

try:
    GPIO.add_event_detect(PIR_PIN, GPIO.RISING, callback=motion_detected, bouncetime=300)
    print("Waiting for motion... Press Ctrl+C to stop.")

    while True:
        time.sleep(1)  # Keeps script running

except RuntimeError as e:
    print(f"? GPIO Error: {e}")

finally:
    cleanup_gpio(None, None)  # Ensures cleanup on exit
