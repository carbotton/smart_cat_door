# override.py
let_in_flag = False

def init_override_button(gpio_pin: int = 27, poll_fallback: bool = True):
    """
    Tries interrupt-based button first.
    If edge detection fails, optionally falls back to polling (no add_event_detect).
    Returns an object you can keep around (or None if disabled).
    """
    global let_in_flag
    try:
        from gpiozero import Button

        btn = Button(gpio_pin, pull_up=True, bounce_time=0.2)
        btn.when_pressed = lambda: _toggle()
        return btn

    except Exception as e:
        print(f"[override] WARNING: interrupt button init failed: {e}")

        if not poll_fallback:
            return None

        # Polling fallback (no edge detection)
        try:
            from gpiozero import Button
            btn = Button(gpio_pin, pull_up=True)
            _start_poll_thread(btn)
            return btn
        except Exception as e2:
            print(f"[override] WARNING: polling fallback failed: {e2}")
            return None


def _toggle():
    global let_in_flag
    let_in_flag = not let_in_flag
    print("LET_IN_FLAG is now", let_in_flag)


def _start_poll_thread(btn, period_s: float = 0.05):
    import threading, time
    def loop():
        last = btn.is_pressed
        while True:
            cur = btn.is_pressed
            if cur and not last:
                _toggle()
            last = cur
            time.sleep(period_s)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
