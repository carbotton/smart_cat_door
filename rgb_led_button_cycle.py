from gpiozero import LED, Button
from signal import pause

led = LED(17)
# pull_up=True means the pin is HIGH by default, goes LOW when pressed
button = Button(18, pull_up=True)

state = False

def toggle_led():
    global state
    state = not state
    if state:
        led.on()
        print("LED ON")
    else:
        led.off()
        print("LED OFF")

button.when_pressed = toggle_led

print("Press the button to toggle the LED.")
pause()
