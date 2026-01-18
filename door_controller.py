# door_controller.py

import RPi.GPIO as GPIO
from config import RELAY_PIN  

LOCKED = GPIO.LOW
UNLOCKED = GPIO.HIGH

# Setup GPIO once
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=LOCKED)


def lock_door():
    """
    Lock door (solenoid ON)
    """
    GPIO.output(RELAY_PIN, LOCKED)

def unlock_door():
    """
    Unlock door (solenoid OFF)
    """
    GPIO.output(RELAY_PIN, UNLOCKED)

def door_cleanup():
    """
    Call this once on shutdown
    """
    GPIO.cleanup()

