# Arduino UNO Q — capture-request button and operator status light

The MCU sketch debounces a D2-to-ground button and exposes a monotonic press
counter through Arduino Router Bridge. The Linux-side Python app polls the
counter, suppresses startup/reboot replay and writes a `capture.requested` JSON
only when camera readiness, brick presence and operator confirmation are all true
in a context file updated within three seconds. A button press never creates a
GOOD/DEFECTIVE result or authorizes robot motion. No board is needed for core tests.

Use Arduino App Lab on an available UNO Q, inspect its wiring/pin-voltage
requirements, and review `hardware.md` before connecting, flashing or configuring
station hardware. The sketch is in `hardware/uno_q/brickcell_button/`. It uses
UNO Q's actual `Arduino_RouterBridge` API and active-low built-in LED. No flashing,
device opening or hardware setup was performed by this commit.

On the UNO Q Linux side, install the Brickcell source in the board's separate App
Lab Python environment and run `python -m brickcell.integrations.arduino --context
/path/to/readiness.json --output /path/to/capture-requests`. The existing capture
UI/controller must supply actual readiness, not a timer-generated heartbeat:

```json
{"object_id":"brick-1","session_id":"session-1","camera_id":"top",
 "camera_ready":true,"object_present":true,"operator_confirmed":true,
 "updated_at":1794657600}
```

The timestamp above is illustrative; stale contexts are deliberately ignored.
Feed saved requests into an operator-reviewed capture workflow. This connector
only queues the request; it does not consume it or start a camera. LED on means
recent capture readiness, blinking means bridge/context error, off means idle.
It does not certify defect absence, grasp success or emergency-stop readiness.
The press counter initializes once per Linux app start and rebases after an MCU
reset. Capture commands and mechanical actions remain outside the firmware.

Offline tests cover startup replay, repeated presses, stale/unconfirmed context
and no robot authorization. Arduino compilation, App Lab runtime, wiring,
debounce timing and physical LEDs remain unverified.
[Arduino UNO Q Router Bridge](https://docs.arduino.cc/tutorials/uno-q/routerbridge-multilanguage/).
