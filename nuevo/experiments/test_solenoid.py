import RPi.GPIO as GPIO
import time

RELAY_PIN = 17  # GPIO pin connected to the relay

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

try:
    print("Turning ON solenoid...")
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # Activate relay (depends on relay type, try LOW if HIGH doesn't work)
    time.sleep(2)  # Keep it ON for 2 seconds
    
    print("Turning OFF solenoid...")
    GPIO.output(RELAY_PIN, GPIO.LOW)  # Deactivate relay
    time.sleep(2)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    GPIO.cleanup()  # Reset GPIO
