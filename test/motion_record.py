import cv2
import lgpio
import time
import signal
import sys
import datetime
import os

PIR_PIN = 26  # Your PIR sensor pin
camera_url = "rtsp://169.254.1.1:554/live/0/MAIN"  # Replace with your camera feed URL
output_dir = "/home/carbotton/camera_feeds"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the GPIO chip
chip = lgpio.gpiochip_open(0)  # Open the first GPIO chip (0 corresponds to gpiochip0)

# Setup for the GPIO pin
lgpio.gpio_claim_input(chip, PIR_PIN)

# Setup video capture
cap = cv2.VideoCapture(camera_url)
if not cap.isOpened():
    print("Error: Could not open the camera feed.")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Function to handle motion detection and start recording
def motion_detected():
    print("?? Motion detected! Starting recording...")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{output_dir}/{timestamp}.avi"
    out = cv2.VideoWriter(filename, fourcc, 20, (frame_width, frame_height))
    
    # Record video while motion is detected
    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            print("Warning: Dropped frame (invalid).")
    
    out.release()  # Save video after stopping the loop
    print(f"Video saved to {filename}.")

# Function to cleanup GPIO when exiting
def cleanup_gpio(signal, frame):
    print("\nCleaning up GPIO and exiting...")
    lgpio.gpiochip_close(chip)  # Close the chip
    cap.release()
    sys.exit(0)

# Register cleanup function for Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, cleanup_gpio)

try:
    print("Waiting for motion... Press Ctrl+C to stop.")

    while True:
        # Polling for motion detection
        if lgpio.gpio_read(chip, PIR_PIN):  # If motion is detected
            motion_detected()
            time.sleep(2)  # Wait to avoid multiple detections
        time.sleep(0.1)  # Sleep for a short time to reduce CPU usage

except RuntimeError as e:
    print(f"? GPIO Error: {e}")

finally:
    cleanup_gpio(None, None)  # Ensures cleanup on exit
