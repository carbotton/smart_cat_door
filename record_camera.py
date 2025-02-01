import cv2
import datetime
import os

# Camera feed URL
camera_url = "rtsp://169.254.1.1:554/live/0/MAIN"

# Output directory
output_dir = "/home/carbotton/camera_feeds"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate filename based on date and time
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{output_dir}/{timestamp}.avi"

# Open the camera feed
cap = cv2.VideoCapture(camera_url)
if not cap.isOpened():
    print("Error: Could not open the camera feed.")
    exit()

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(filename, fourcc, 20, (frame_width, frame_height))

print(f"Saving video to {filename}. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)  # Write valid frames to the video file
        else:
            print("Warning: Dropped frame (invalid).")
except KeyboardInterrupt:
    print("\nStopped recording.")
finally:
    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {filename}.")
