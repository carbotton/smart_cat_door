import RPi.GPIO as GPIO
import time

# Define the GPIO pin for the PIR sensor
PIR_PIN = 26  # Change this if your sensor is connected to a different GPIO pin

# Setup GPIO
GPIO.setwarnings(False)  # Ignore warnings
GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbering
GPIO.setup(PIR_PIN, GPIO.IN)  # Set PIR pin as input

print("PIR Sensor Test. Waiting for motion...")

try:
    while True:
        if GPIO.input(PIR_PIN):  # Motion detected
            print("🚨 Motion detected!")
        else:  # No motion
            print("No movement.")
        
        time.sleep(1)  # Delay to avoid flooding the console

except KeyboardInterrupt:
    print("\nExiting...")
    GPIO.cleanup()  # Reset GPIO settings
