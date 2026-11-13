# Arm API service (armd) — instructions for agents

This document tells you how to use, run, deploy, and change **armd**. armd is an HTTP API service for the two SO-101 robot arms on the Intel server.

> For SSH access, read [ssh.md](ssh.md) first. For the hardware and the software setup, read [hardware.md](hardware.md).

## 1. Purpose

armd controls the two SO-101 arms through one HTTP API. It does these tasks:

- It reads the state of all servos (position, speed, load, voltage, temperature, current, errors) many times each second.
- It moves the joints of the follower arm to a position, at a speed and acceleration that you set.
- It applies safety limits to all moves (see [Safety rules](#5-safety-rules)).
- It closes and opens the serial ports on request, so that LeRobot can use the arms.

The service runs on the Intel server. It listens on `127.0.0.1:8765` only. Use an SSH tunnel to get access from your laptop.

## 2. Where the code is

All code is in [arm-api/](arm-api/) in this repo. On the server, the code is in `/home/intel/arm-api`.

| File | What it does |
|---|---|
| `arm-api/pyproject.toml` | Python package data and dependencies (`ftservo-python-sdk`, `pyserial`, `fastapi`, `uvicorn`, `pyyaml`). |
| `arm-api/config.yaml` | Configuration of the server and the two arms. The service reads it at start. |
| `arm-api/armd/__init__.py` | Package version (`__version__`). |
| `arm-api/armd/__main__.py` | Command line entry point: `python -m armd [--config PATH] [--fake]`. |
| `arm-api/armd/types.py` | Shared constants (tick and speed limits), unit conversion, the `ServoReading` record, the error classes, and the `BusProtocol` interface. |
| `arm-api/armd/servo.py` | Low-level STS3215 servo code (from commanderfun/STS3215, MIT license). |
| `arm-api/armd/registers.py` | STS3215 register map, and decode of sign-magnitude values. |
| `arm-api/armd/bus.py` | `Bus`: one serial port with the servos on it. Real hardware. |
| `arm-api/armd/fake.py` | `FakeBus`: simulated servos. Use it for tests and for work without hardware. |
| `arm-api/armd/config.py` | Reads `config.yaml` into Python dataclasses. |
| `arm-api/armd/arm.py` | `Arm`: safety rules, limits (clamps), and the poller thread. |
| `arm-api/armd/server.py` | FastAPI app (`create_app`). All HTTP and WebSocket endpoints. |
| `arm-api/armd/client.py` | `ArmClient`: Python client. It uses the standard library only. |
| `arm-api/tests/` | Unit tests (`test_bus.py`, `test_arm.py`, `test_server.py`). They use a mocked SDK or `FakeBus`. |
| `arm-api/deploy/armd.service` | systemd unit for the service. |
| `arm-api/deploy/deploy.sh` | Runs on the laptop. Copies the code to the server and installs it into the `armd` conda env. |
| `arm-api/deploy/install.sh` | Runs on the server. Checks the setup and shows the `sudo` commands that install or remove the systemd unit. |

To run the tests on the laptop:

```bash
cd arm-api && uv run --python 3.12 --extra dev pytest -q
```

## 3. The hardware that armd controls

Each arm has six Feetech STS3215 servos on one serial bus (1 000 000 baud). Each arm connects to the server with a USB serial adapter (CH340, `1a86`).

| Arm | Power | Port (stable name) | Can move? |
|---|---|---|---|
| `follower` | 12 V | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00` | Yes (`allow_motion: true`) |
| `leader` | 5 V | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044736-if00` | No. Read-only. You move it by hand for teleoperation. |

Always use the `/dev/serial/by-id/...` names. The `/dev/ttyACM*` numbers can change after a reconnect.

Both arms have the same joints:

| Joint name | Servo ID |
|---|---|
| `shoulder_pan` | 1 |
| `shoulder_lift` | 2 |
| `elbow_flex` | 3 |
| `wrist_flex` | 4 |
| `wrist_roll` | 5 |
| `gripper` | 6 |

## 4. Units

| Quantity | Unit | Range | Notes |
|---|---|---|---|
| Position | ticks | 0–4095 | One turn = 4096 ticks. 2048 = 180° (the center). |
| Degrees | degrees | 0–360 | `degrees = ticks * 360 / 4096`. 0° is not the center. 180° is the center. |
| Speed | ticks/s | 1–3400 | The API clamps it to the arm `max_speed`. 0 is not permitted (see below). |
| Acceleration (`acc`) | servo unit | 0–254 | 0 = no ramp. A larger value gives a faster ramp. |
| Load | percent | -100–100 | Signed. |
| Voltage | volts | — | |
| Temperature | °C | — | |
| Current | mA | — | |
| `delta` | ticks | — | Relative move from the present position. |
| `delta_degrees` | degrees | — | Relative move from the present position. |
| Timestamps | Unix time, seconds | — | `last_poll`, `timestamp`. |

## 5. Safety rules

armd applies these rules to all requests:

- **Clamps.** Each joint has a `min` and a `max` in ticks. They come from `config.yaml`, or else from the servo EEPROM (read at start). A target outside the limits is **clamped** to the limit. The request does not fail. The response has `"clamped": true`.
- **Speed clamp.** All speeds are clamped to the arm `max_speed`.
- **Speed 0 is refused** (400 `invalid_request`). On the servo, speed 0 means "maximum speed". This is not safe.
- **Motion permission.** Moves and torque **on** need `allow_motion: true` for the arm. Else the API gives 403 `motion_disabled`.
- **The leader arm is read-only.** It has `allow_motion: false`. You can read it. You cannot move it or set its torque on.
- **Torque off and stop are always permitted**, also on a read-only arm.
- **block_on_errors.** If the last reading of a joint has an error in `block_on_errors` (default: `overload`, `temperature`, `current`, `sensor`), the API refuses moves of that joint (409 `joint_fault`).
- **Torque on before a move.** A move turns on the torque of the joints that it moves. It also sets position mode if the joint is in another mode.
- **Stop.** `POST /arms/{arm}/stop` with `"mode": "hold"` stops all joints at the present position. `"mode": "limp"` turns off the torque of all joints. **With limp, the arm falls.** Hold the arm before you send limp.
- **on_shutdown.** When the service stops, `hold` does nothing (the servos keep their torque and position). `limp` turns off the torque of all joints.
- **Wheel mode is off.** Wheel mode (continuous rotation, `/spin`) needs `allow_motion: true` **and** `allow_spin: true`. Both arms have `allow_spin: false`. Arm joints must not turn continuously. Keep it off.
- **Admin endpoints are off.** The raw register endpoints need `server.admin: true` or `ARMD_ADMIN=1`. Else they give 403. Writes to EEPROM (address < 40) and to the lock register (address 55) are always refused (403).
- **Localhost only.** The service listens on `127.0.0.1`. It has no authentication. Do not change `host` to `0.0.0.0`.

## 6. Configuration

The service reads the configuration file once, at start. After you change the file, restart the service.

The service finds the file in this order:

1. The `--config PATH` argument.
2. The `ARMD_CONFIG` environment variable.
3. `arm-api/config.yaml` (next to the `armd` package).

### `server` keys

| Key | Default | What it does |
|---|---|---|
| `host` | `127.0.0.1` | Address to listen on. Keep localhost only. |
| `port` | `8765` | TCP port. |
| `poll_hz` | `30` | How many times each second the poller reads the state of each arm. |
| `admin` | `false` | `true` turns on the raw register endpoints. |

### `arms.<arm>` keys

| Key | Example | What it does |
|---|---|---|
| `port` | `/dev/serial/by-id/usb-1a86_...-if00` | Serial port of the arm. |
| `baudrate` | `1000000` | Serial speed. The servos use 1 000 000. |
| `allow_motion` | `true` / `false` | `false` = read-only arm. Torque off and stop are still permitted. |
| `allow_spin` | `false` | `true` permits wheel mode (`/spin`). Keep `false`. |
| `max_speed` | `1000` | Maximum speed in ticks/s. The API clamps all speeds to this value. |
| `default_speed` | `300` | Speed (ticks/s) for a request that gives no speed. |
| `default_acc` | `30` | Acceleration (0–254) for a request that gives no `acc`. |
| `on_shutdown` | `hold` | `hold` = do nothing at stop. `limp` = torque off for all joints at stop. |
| `block_on_errors` | `[overload, temperature, current, sensor]` | Servo errors that block moves of that joint. Possible names: `voltage`, `sensor`, `temperature`, `current`, `overload`. |
| `joints` | see below | Joint names and their settings. |

### `arms.<arm>.joints.<joint>` keys

| Key | Required | What it does |
|---|---|---|
| `id` | Yes | Servo ID on the bus (1–6). |
| `min` | No | Lowest permitted position (ticks). Overrides the EEPROM limit. |
| `max` | No | Highest permitted position (ticks). Overrides the EEPROM limit. |
| `default_speed` | No | Default speed for this joint. Overrides the arm `default_speed`. |
| `default_acc` | No | Default acceleration for this joint. Overrides the arm `default_acc`. |

Example: limit the gripper and make it slower.

```yaml
      gripper:       {id: 6, min: 1900, max: 3300, default_speed: 200}
```

### Environment variables

| Variable | What it does |
|---|---|
| `ARMD_CONFIG` | Path of the configuration file (if there is no `--config`). |
| `ARMD_FAKE=1` | Use simulated servos (`FakeBus`) for all arms. Same as `--fake`. |
| `ARMD_ADMIN=1` | Turn on the raw register endpoints. Same as `server.admin: true`. |

## 7. How to run the service

Run only **one** instance of armd at a time. Two instances cannot use the same serial ports or the same TCP port.

### 7.1 Deploy the code (from the laptop)

```bash
arm-api/deploy/deploy.sh            # Copy the code, create the env if necessary, pip install.
arm-api/deploy/deploy.sh --dry-run  # Show what rsync copies. Do not change the server.
```

The script copies `arm-api/` to `/home/intel/arm-api/`. Then it runs `pip install -e /home/intel/arm-api` in the conda env `armd` (Python 3.12, `/home/intel/miniforge3/envs/armd`). You can run it again at any time. It does not restart the service. After a deploy, restart the service (see 7.4).

**Do not install armd into the `lerobot` env.** The armd SDK (`ftservo-python-sdk`) conflicts with the LeRobot SDK (`feetech-servo-sdk`).

### 7.2 Manual run in tmux

Use this when the systemd service is not installed or is stopped.

```bash
ssh intel
tmux -L armd new -d -s armd 'cd /home/intel/arm-api && ~/miniforge3/envs/armd/bin/python -m armd --config /home/intel/arm-api/config.yaml 2>&1 | tee ~/armd.log'
tmux -L armd attach -t armd  # See the output. Push Ctrl-b, then d, to detach.
tmux -L armd kill-session -t armd  # Stop the service.
```

Use `-L armd` (a separate tmux server). A tmux server keeps the groups that it had when it started. The default tmux server on the Intel server started before `intel` got the `dialout` group, so a service in that server gets "permission denied" on the ports.

### 7.3 Run with no hardware (`--fake`)

`--fake` uses simulated servos. It does not open a serial port. Use it on the laptop or on the server.

```bash
cd arm-api && uv run --python 3.12 python -m armd --fake
# or on the server:
~/miniforge3/envs/armd/bin/python -m armd --config /home/intel/arm-api/config.yaml --fake
```

### 7.4 Run as a systemd service

The `intel` user can use `sudo` without a password (see [ssh.md](ssh.md#sudo)). With no option, the install script only shows the root commands. With `--run`, it runs them with `sudo -n`.

Install:

```bash
ssh intel
~/arm-api/deploy/install.sh            # Checks the setup, then shows the sudo commands.
~/arm-api/deploy/install.sh --run      # Checks the setup, then runs the sudo commands.
```

The script shows these commands:

```bash
sudo -n install -m 644 /home/intel/arm-api/deploy/armd.service /etc/systemd/system/armd.service && sudo -n systemctl daemon-reload && sudo -n systemctl enable --now armd
```

Operate the service:

| Task | Command |
|---|---|
| Show the status | `systemctl status armd` or `~/arm-api/deploy/install.sh status` |
| Follow the log | `journalctl -u armd -f` |
| Show the last 100 log lines | `journalctl -u armd -n 100 --no-pager` |
| Restart (after a deploy or a config change) | `sudo systemctl restart armd` |
| Stop | `sudo systemctl stop armd` |
| Start | `sudo systemctl start armd` |
| Remove the service | `~/arm-api/deploy/install.sh uninstall --run` (without `--run`, it only shows the sudo commands) |

The unit has `Restart=always`. If the process stops, systemd starts it again after 2 s. To stop the service, use `systemctl stop`. Do not use `kill`.

If `journalctl` shows no lines, your user cannot read the system journal. Use `sudo journalctl -u armd -f`.

## 8. How to get access

The service listens on `127.0.0.1:8765` on the server only.

- **On the server:** use `http://127.0.0.1:8765`.
- **From the laptop:** open an SSH tunnel. Then use `http://localhost:8765` on the laptop.

  ```bash
  ssh -L 8765:localhost:8765 intel          # Tunnel and a shell.
  ssh -N -L 8765:localhost:8765 intel       # Tunnel only, no shell.
  ```

- **Interactive API docs:** open `http://localhost:8765/docs` in a browser. The OpenAPI schema is at `/openapi.json`.

## 9. Endpoint reference

Base URL: `http://127.0.0.1:8765`. All bodies are JSON. The examples use these shell variables:

```bash
B=http://127.0.0.1:8765
F=$B/arms/follower
J='Content-Type: application/json'
```

`{arm}` is `follower` or `leader`. `{joint}` is a joint name from [section 3](#3-the-hardware-that-armd-controls).

While an arm is released (see `/release`), calls that use the bus give 409 `bus_released`. These calls still work: `GET /health`, `GET /arms`, `GET /arms/{arm}`, `POST /arms/{arm}/acquire`, and the cache reads `GET /arms/{arm}/state` and `GET /arms/{arm}/joints/{joint}` without `fresh=true`, and the WebSocket stream. The cache reads show status `released` and null readings. In a `/positions` body, an unknown joint name gives 404 `not_found`, and an unknown field gives 400 `invalid_request`.

### 9.1 The JointState object

Many endpoints send this object for each joint:

```json
{
  "name": "elbow_flex", "id": 3,
  "position": 2051, "degrees": 180.26, "goal_position": 2048,
  "speed": 0, "goal_speed": 300, "acceleration": 30,
  "load": 4.2, "voltage": 12.1, "temperature": 31, "current_ma": 18.0,
  "moving": false, "torque_enabled": true, "mode": 0,
  "errors": [], "healthy": true,
  "min": 800, "max": 3300, "default_speed": 300, "default_acc": 30,
  "online": true
}
```

- `mode`: 0 = position, 1 = wheel, 2 = PWM, 3 = step.
- `errors`: names of the active servo errors: `voltage`, `sensor`, `temperature`, `current`, `overload`.
- `healthy`: `true` if the joint has no blocking error.
- `online: false`: the servo did not answer at the last poll. Then the reading fields are `null`.

### 9.2 GET /health

Service status. `"status": "ok"` if all arms are online. Else `"degraded"`.

```bash
curl -s $B/health
```

```json
{"status": "degraded", "version": "0.1.0", "admin": false,
 "arms": {
   "follower": {"status": "online", "port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00", "last_poll": 1763070000.12, "error": null},
   "leader": {"status": "offline", "port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044736-if00", "last_poll": null, "error": "port not found"}}}
```

Arm `status` is `online`, `offline`, or `released`.

### 9.3 GET /arms and GET /arms/{arm}

Static data about the arms. `/arms` sends a list. `/arms/{arm}` sends one object.

```bash
curl -s $B/arms
curl -s $F
```

```json
{"name": "follower", "port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00",
 "status": "online", "allow_motion": true, "allow_spin": false,
 "max_speed": 1000, "default_speed": 300, "default_acc": 30,
 "joints": [{"name": "shoulder_pan", "id": 1, "min": 0, "max": 4095}, "..."]}
```

### 9.4 GET /arms/{arm}/state?fresh=false

State of all joints. By default it sends the last poller reading. `fresh=true` reads the bus now.

```bash
curl -s "$F/state"
curl -s "$B/arms/leader/state?fresh=true"
```

```json
{"arm": "follower", "status": "online", "timestamp": 1763070000.12, "age_ms": 14,
 "joints": {"shoulder_pan": {"name": "shoulder_pan", "id": 1, "position": 2047, "...": "..."}, "...": {}}}
```

`age_ms` is the age of the reading in milliseconds.

### 9.5 GET /arms/{arm}/joints/{joint}?fresh=false

State of one joint (a JointState object).

```bash
curl -s "$F/joints/gripper"
```

### 9.6 POST /arms/{arm}/joints/{joint}/position

Move one joint. Give **exactly one** of these target fields:

| Field | Type | Meaning |
|---|---|---|
| `position` | int | Absolute target in ticks. |
| `degrees` | float | Absolute target in degrees. |
| `delta` | int | Relative move in ticks. |
| `delta_degrees` | float | Relative move in degrees. |

Optional fields:

| Field | Default | Meaning |
|---|---|---|
| `speed` | joint, then arm `default_speed` | ticks/s, 1–3400, clamped to `max_speed`. |
| `acc` | joint, then arm `default_acc` | 0–254. |
| `wait` | `false` | `true` = wait until the joint gets to the target, or until `timeout`. |
| `timeout` | `5.0` | Seconds to wait (with `wait: true`). |
| `tolerance` | `10` | Ticks. The joint "got to the target" if it is this near. |

```bash
curl -s -X POST -H "$J" $F/joints/elbow_flex/position -d '{"position": 2300, "speed": 200}'
curl -s -X POST -H "$J" $F/joints/wrist_roll/position -d '{"degrees": 90, "wait": true}'
curl -s -X POST -H "$J" $F/joints/gripper/position  -d '{"delta": -100}'
```

```json
{"joint": "wrist_roll", "target": 1024, "clamped": false, "speed": 300, "acc": 30,
 "reached": true, "final_position": 1027}
```

With `wait: false`, `reached` and `final_position` are `null`. If the timeout ends first, `reached` is `false`. This is not an error.

### 9.7 POST /arms/{arm}/positions

Move many joints with one sync write. All joints start at the same time. Each joint takes one target field (as in 9.6), and optional `speed` and `acc`. The top-level `speed` and `acc` apply to joints that do not give their own.

```bash
curl -s -X POST -H "$J" $F/positions -d '{
  "joints": {
    "shoulder_lift": {"position": 1800},
    "elbow_flex":    {"degrees": 200, "speed": 150},
    "gripper":       {"delta": 50}
  },
  "speed": 250, "acc": 20, "wait": true, "timeout": 6.0, "tolerance": 10
}'
```

```json
{"targets": {
   "shoulder_lift": {"target": 1800, "clamped": false, "speed": 250, "acc": 20},
   "elbow_flex":    {"target": 2276, "clamped": false, "speed": 150, "acc": 20},
   "gripper":       {"target": 3300, "clamped": true,  "speed": 250, "acc": 20}},
 "reached": true,
 "final_positions": {"shoulder_lift": 1803, "elbow_flex": 2274, "gripper": 3296}}
```

With `wait: false`, `reached` and `final_positions` are `null`.

### 9.8 PUT /arms/{arm}/joints/{joint}/speed

Set the default speed (and optional default acceleration) of one joint. The change stays in memory only. It does not change `config.yaml`. A restart removes it.

```bash
curl -s -X PUT -H "$J" $F/joints/shoulder_pan/speed -d '{"speed": 150, "acc": 10}'
```

```json
{"joint": "shoulder_pan", "default_speed": 150, "default_acc": 10}
```

### 9.9 POST /arms/{arm}/joints/{joint}/spin

Wheel mode (continuous rotation). Needs `allow_motion` **and** `allow_spin`. Both are off for the arms, so this endpoint gives 403. `speed` is -3400 to 3400. The API clamps `|speed|` to `max_speed`. Speed 0 stops the joint. A later position move sets position mode again.

```bash
curl -s -X POST -H "$J" $F/joints/wrist_roll/spin -d '{"speed": 200}'
curl -s -X POST -H "$J" $F/joints/wrist_roll/spin -d '{"speed": 0}'
```

```json
{"joint": "wrist_roll", "speed": 200, "acc": 30, "mode": 1}
```

### 9.10 POST /arms/{arm}/torque

Turn the torque on or off for some joints or all joints. No `joints` field = all joints. Torque **off** is always permitted. Torque **on** needs `allow_motion`.

**Caution:** with torque off, the arm falls. Hold the arm first.

```bash
curl -s -X POST -H "$J" $F/torque -d '{"enabled": false, "joints": ["gripper"]}'
curl -s -X POST -H "$J" $F/torque -d '{"enabled": true}'
```

```json
{"torque": {"gripper": false}}
```

### 9.11 POST /arms/{arm}/joints/{joint}/torque

Torque on or off for one joint. Same rules as 9.10.

```bash
curl -s -X POST -H "$J" $F/joints/gripper/torque -d '{"enabled": false}'
```

```json
{"torque": {"gripper": false}}
```

### 9.12 POST /arms/{arm}/stop

Stop all joints. Always permitted (also on the leader arm).

- `"mode": "hold"` (default): set the goal of each joint to its present position. The arm stays in place.
- `"mode": "limp"`: torque off for all joints. **The arm falls.** `positions` are the last readings.

```bash
curl -s -X POST -H "$J" $F/stop -d '{}'
curl -s -X POST -H "$J" $F/stop -d '{"mode": "limp"}'
```

```json
{"mode": "hold", "positions": {"shoulder_pan": 2047, "shoulder_lift": 1803, "elbow_flex": 2274,
 "wrist_flex": 2100, "wrist_roll": 1027, "gripper": 3296}}
```

### 9.13 POST /arms/{arm}/release and POST /arms/{arm}/acquire

`/release` closes the serial port of the arm. The poller pauses. Then another program (for example LeRobot) can open the port. `/acquire` opens the port again.

```bash
curl -s -X POST $F/release
curl -s -X POST $F/acquire
```

```json
{"arm": "follower", "status": "released"}
{"arm": "follower", "status": "online"}
```

The release does not change the torque. With `on_shutdown: hold`, the servos keep their torque.

### 9.14 Raw registers (admin only)

These endpoints need `server.admin: true` or `ARMD_ADMIN=1`. Else they give 403. Writes to EEPROM (address < 40) and to the lock register (address 55) are always refused (403). `size` is 1 or 2 (bytes).

```bash
curl -s "$F/joints/gripper/registers/56?size=2"                              # Read present position.
curl -s -X PUT -H "$J" $F/joints/gripper/registers/41 -d '{"value": 20, "size": 1}'  # Write acceleration.
```

```json
{"addr": 56, "size": 2, "value": 3296}
```

### 9.15 WebSocket /arms/{arm}/stream?hz=10

Sends the `/state` JSON object `hz` times each second (1–50) until the client closes the connection.

```bash
websocat "ws://127.0.0.1:8765/arms/leader/stream?hz=20"
```

Python (the `websockets` package is in the `armd` env):

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8765/arms/leader/stream?hz=20") as ws:
        while True:
            state = json.loads(await ws.recv())
            print(state["joints"]["gripper"]["position"])

asyncio.run(main())
```

## 10. Error codes

All errors have this body:

```json
{"error": "<code>", "detail": "<text>"}
```

| HTTP status | `error` | Cause |
|---|---|---|
| 400 | `invalid_request` | Bad value or bad body. Examples: speed 0, no target field or two target fields, a value out of range, a bad JSON body. |
| 403 | `motion_disabled` | The arm has `allow_motion: false` (move or torque on), `allow_spin: false` (spin), or admin is off (registers). Also an EEPROM or lock register write. |
| 404 | `not_found` | Unknown arm or joint. |
| 409 | `joint_fault` | The joint has an error in `block_on_errors`. The move is refused. |
| 409 | `bus_released` | The arm is released. Send `/acquire` first. |
| 500 | `internal_error` | Bug in armd. Read the log. |
| 502 | `servo_error` | A servo did not answer, or sent an error packet. |
| 503 | `bus_error` | The port is missing or closed, or the link failed. |

## 11. Python client

`arm-api/armd/client.py` uses the Python standard library only. You can use it in any Python 3.11+ environment. You can also copy the file.

```python
import sys; sys.path.insert(0, "arm-api")      # Or: pip install -e arm-api
from armd.client import ArmClient, ArmClientError

follower = ArmClient()                          # base_url="http://127.0.0.1:8765", arm="follower", timeout=10
leader = ArmClient(arm="leader")

print(follower.health()["status"])
print(follower.arms())

# Read the state.
state = follower.state()                        # Last poller reading.
pos = follower.joint("gripper", fresh=True)["position"]

# Move one joint and wait.
r = follower.move("elbow_flex", degrees=200, speed=200, wait=True, timeout=5.0)
print(r["reached"], r["final_position"], r["clamped"])

# Relative move.
follower.move("gripper", delta=-100)

# Move many joints with one sync write.
follower.move_many({"shoulder_lift": {"position": 1800},
                    "elbow_flex": {"degrees": 190, "speed": 150}},
                   speed=250, wait=True)

# Copy the leader pose to the follower one time.
pose = {name: {"position": j["position"]}
        for name, j in leader.state()["joints"].items() if j["online"]}
follower.move_many(pose, speed=300, wait=True)

# Default speed (memory only), torque, stop.
follower.set_speed("shoulder_pan", 150, acc=10)
follower.torque(False, joints=["gripper"])      # Hold the arm first.
follower.stop()                                 # mode="hold"
follower.stop(mode="limp")                      # The arm falls.

# Errors.
try:
    leader.move("gripper", position=2000)
except ArmClientError as e:
    print(e.status, e.code, e.detail)           # 403 motion_disabled ...

# Give the port to LeRobot, then take it back.
follower.release()
follower.acquire()
```

Other methods: `spin(joint, speed, acc=None)`, `read_register(joint, addr, size=1)`, `write_register(joint, addr, value, size=1)`.

## 12. LeRobot and armd

Only **one** process can open a serial port at a time. If armd has the port, LeRobot cannot open it. If LeRobot has the port, armd shows the arm as `offline`.

Before you start LeRobot (calibration, teleoperation, recording), do one of these:

- **Release the arms** (armd continues to run):

  ```bash
  curl -s -X POST http://127.0.0.1:8765/arms/follower/release
  curl -s -X POST http://127.0.0.1:8765/arms/leader/release
  ```

  After LeRobot stops, acquire the arms again:

  ```bash
  curl -s -X POST http://127.0.0.1:8765/arms/follower/acquire
  curl -s -X POST http://127.0.0.1:8765/arms/leader/acquire
  ```

- **Stop the service:** `sudo systemctl stop armd`. After LeRobot stops: `sudo systemctl start armd`. For a manual run, use `tmux -L armd kill-session -t armd`.

Use the `lerobot` conda env for LeRobot. Use the `armd` env for armd. Do not mix them.

## 13. Troubleshooting

| Problem | What to do |
|---|---|
| An arm is `offline`, and the error tells that the port is missing | Run `ls -l /dev/serial/by-id`. If the port is not there, make sure that the USB cable is connected and that the USB-C hub has power. Read the kernel log: `journalctl -k --since "10 min ago" \| grep -i usb`. armd tries to open the port again every 2 s. You do not have to restart it. |
| `Permission denied` on the port | The user must be in the `dialout` group: `id -nG intel`. The systemd unit sets `Group=dialout`. For a manual run, log out and log in again after a group change. An old tmux server also keeps old groups: use `tmux -L armd` (see 7.2). |
| `Device or resource busy`, or the arm is `offline` while LeRobot runs | Another process has the port. Find it: `fuser -v /dev/ttyACM0 /dev/ttyACM1`. Stop LeRobot, or release the arm in armd (see section 12). |
| The arm is `online`, but one joint has `online: false` | That servo did not answer. Examine the cable between the servos (daisy chain) and the servo ID. |
| All joints of the follower do not answer | Make sure that the 12 V power supply of the follower is on. The USB cable does not give power to the servos. |
| All joints of the leader do not answer | Make sure that the 5 V power supply of the leader is connected. |
| 409 `bus_released` | The arm was released. Send `POST /arms/{arm}/acquire`. |
| 409 `joint_fault` | Read `errors` in the joint state. `overload`: remove the load or the blockage. `temperature`: stop and let the servo become cool. `current`: examine the load and the power supply. `sensor`: examine the servo. Then turn the torque off and on again for that joint, and read the state again. |
| 403 `motion_disabled` on the leader | This is correct. The leader is read-only. |
| 400 for speed 0 | This is correct. Use a speed from 1 to `max_speed`. |
| `Address already in use` (port 8765) | armd runs already (as a service or in tmux). Use the instance that runs, or stop it first. |
| The service restarts again and again | Read the log: `journalctl -u armd -n 100 --no-pager`. Make sure that the `armd` env exists and that `deploy.sh` ran. |
| No answer on `localhost:8765` from the laptop | Open the SSH tunnel (section 8). Make sure that the service runs: `systemctl status armd`. |
