# door_controller.py
import RPi.GPIO as GPIO
from config import RELAY_PIN

LOCKED = GPIO.LOW
UNLOCKED = GPIO.HIGH

_initialized = False

def init_gpio():
    global _initialized
    if _initialized:
        return
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_PIN, GPIO.OUT, initial=LOCKED)
    _initialized = True

def lock_door():
    init_gpio()
    GPIO.output(RELAY_PIN, LOCKED)

def unlock_door():
    init_gpio()
    GPIO.output(RELAY_PIN, UNLOCKED)

def door_cleanup():
    if _initialized:
        GPIO.cleanup()
