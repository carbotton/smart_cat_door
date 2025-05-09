from gpiozero import Button, LED
from signal import pause
import subprocess

# GPIO pins
shutdown_button = Button(18, pull_up=True, bounce_time=0.2)
toggle_button = Button(27, pull_up=True, bounce_time=0.2)
led = LED(17)

# Start with flag = True (LED off)
let_in_flag = True
led.on()

def shutdown_system():
    print("Shutting down...")
    subprocess.run(['sudo', 'shutdown', 'now'])

def toggle_flag():
    global let_in_flag
    let_in_flag = not let_in_flag
    print(f"Flag is now: {let_in_flag}")
    update_led()

def update_led():
    if let_in_flag:
        print("Cat is allowed to go in :)")
        led.on()   # actually turns OFF in your current wiring
    else:
        print("Cat is NOT allowed to go in >:(")
        led.off()  # actually turns ON in your current wiring


# Set up button actions
shutdown_button.when_pressed = shutdown_system
toggle_button.when_pressed = toggle_flag

# Initialize LED based on initial flag value
update_led()

print("Press BLACK button (GPIO18) to shut down the Raspberry Pi.")
print("Press YELLOW button (GPIO27) to toggle LET_IN_FLAG and LED.")
pause()
