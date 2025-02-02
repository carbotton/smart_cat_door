import cv2
import datetime
import os
import time

# Camera feed URL
camera_url = "rtsp://169.254.1.1:554/live/0/MAIN"

# Output directory
output_dir = "/home/carbotton/camera_feeds"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to generate a new filename based on date and time
def generate_filename():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{output_dir}/{timestamp}.avi"

# Open the camera feed
cap = cv2.VideoCapture(camera_url)
if not cap.isOpened():
    print("Error: Could not open the camera feed.")
    exit()

# Video codec settings
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the first video file
filename = generate_filename()
out = cv2.VideoWriter(filename, fourcc, 20, (frame_width, frame_height))
print(f"Saving video to {filename}. Press Ctrl+C to stop.")

start_time = time.time()  # Track when the current video started

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Dropped frame (invalid).")
            continue

        out.write(frame)  # Write valid frames to the video file

        # Check if 15 minutes have passed
        if time.time() - start_time >= 900:  # 900 seconds = 15 minutes
            print("\nCreating a new video file...")

            # Close the current file
            out.release()

            # Generate a new filename and start a new recording
            filename = generate_filename()
            out = cv2.VideoWriter(filename, fourcc, 20, (frame_width, frame_height))
            print(f"New file: {filename}")

            start_time = time.time()  # Reset timer

except KeyboardInterrupt:
    print("\nStopped recording.")

finally:
    # Release resources
    cap.release()
    out.release()
    print(f"Last video saved to {filename}.")
