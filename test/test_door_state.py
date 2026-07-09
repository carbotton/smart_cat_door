"""Door state machine checks — runs off-Pi (RPi.GPIO stubbed).

    .venv/bin/python test/test_door_state.py
"""
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub RPi.GPIO so door_controller imports on a dev machine
gpio = types.ModuleType("RPi.GPIO")
gpio.LOW, gpio.HIGH, gpio.BCM, gpio.OUT = 0, 1, 11, 0
for fn in ("setwarnings", "setmode", "setup", "output", "cleanup"):
    setattr(gpio, fn, lambda *a, **k: None)
rpi = types.ModuleType("RPi")
rpi.GPIO = gpio
sys.modules["RPi"] = rpi
sys.modules["RPi.GPIO"] = gpio

import main
import override


def reset():
    main.door_locked = True
    main.lock_until_ts = None
    main.unlock_until_ts = None
    main.last_event_nr = None
    main.clean_hits_this_event = 0
    override.let_in_flag = False


# startup default: locked
reset()
assert main.door_locked

# clean unlock, then re-lock when window expires
reset()
main.door_decision_cb("no_prey", event_nr=1)
assert main.door_locked, "1 clean hit must not unlock"
main.door_decision_cb("no_prey", event_nr=1)
assert not main.door_locked, "2 clean hits must unlock"
main.timer_tick(now=time.time() + 1)
assert not main.door_locked, "must stay unlocked inside window"
main.timer_tick(now=time.time() + main.UNLOCK_DURATION_SECONDS + 1)
assert main.door_locked, "must re-lock after unlock window"

# prey during unlock window -> immediate lock
reset()
main.door_decision_cb("no_prey", event_nr=2)
main.door_decision_cb("no_prey", event_nr=2)
assert not main.door_locked
main.door_decision_cb("prey", event_nr=2)
assert main.door_locked and main.unlock_until_ts is None

# prey lock window: decisions ignored, expiry returns to locked
reset()
main.door_decision_cb("prey", event_nr=3)
assert main.door_locked and main.lock_until_ts is not None
main.door_decision_cb("no_prey", event_nr=3)
main.door_decision_cb("no_prey", event_nr=3)
assert main.door_locked, "clean hits inside prey lock must be ignored"
main.timer_tick(now=time.time() + main.LOCK_DURATION_SECONDS + 1)
assert main.door_locked and main.lock_until_ts is None

# override on -> unlock; override off -> re-lock
reset()
override.let_in_flag = True
main.timer_tick()
assert not main.door_locked, "override must unlock"
override.let_in_flag = False
main.timer_tick()
assert main.door_locked, "override off must re-lock"

# dk keeps state
reset()
main.door_decision_cb("no_prey", event_nr=4)
main.door_decision_cb("no_prey", event_nr=4)
main.door_decision_cb("dk", event_nr=4)
assert not main.door_locked

print("all door state checks passed")
