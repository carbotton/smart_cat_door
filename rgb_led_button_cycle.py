from gpiozero import Button
import subprocess
from signal import pause

# Set up the button
button = Button(18, pull_up=True, bounce_time=0.2)

# Function to shutdown the system
def shutdown_system():
    print("Shutting down...")
    subprocess.run(['sudo', 'shutdown', 'now'])

# Set up the button press action
button.when_pressed = shutdown_system

print("Press the button to shut down the system.")
pause()  # Wait for the button press
