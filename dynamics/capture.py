"""Bounded follower trajectory capture through armd; explicit hardware opt-in.

Each joint is excited separately within the explicitly selected small envelope.
All targets stay within the existing calibration intervals. Finalization
always requests hold; torque is never disabled. No detector is involved.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np

from model import MOTOR_NAMES, ROOT, ServoCalibration, digest, dump


class API:
    def __init__(self):
        self.base = "http://127.0.0.1:8765/arms/follower"

    def call(self, path, body=None):
        request = urllib.request.Request(
            self.base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=0.75) as response:
            return json.load(response)


def check_state(state, center, deviation_ticks=57):
    if state["status"] != "online" or state["age_ms"] > 200:
        raise RuntimeError("Follower state is unavailable or stale")
    for i, name in enumerate((*MOTOR_NAMES, "gripper")):
        r = state["joints"][name]
        if not r["online"] or r["position"] is None:
            raise RuntimeError(f"Untrusted joint feedback: {name}")
        if i < 5 and abs(r["position"] - center[i]) > deviation_ticks:
            raise RuntimeError(f"Joint left selected monitoring envelope: {name}")


def capture(
    output,
    partition,
    pid_snapshot,
    amplitude=2,
    period=6,
    speed=None,
):
    if partition not in ("fit", "val", "test") or not 0 < amplitude <= 20 or period < 5:
        raise ValueError("Unsupported capture envelope")
    delta = round(amplitude * 4096 / 360)
    if speed is not None and not 1 <= speed <= 3400:
        raise ValueError(
            "Explicit recording speed must be nonzero; API clamps to its configured maximum"
        )
    deviation_ticks = delta + round(3 * 4096 / 360)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    api = API()
    calibration = ServoCalibration("follower")
    gains = json.loads(Path(pid_snapshot).read_text())
    if set(gains["joints"]) != set((*MOTOR_NAMES, "gripper")):
        raise RuntimeError("PID snapshot must cover all six follower servos")
    if any(row["I"] != 2 or row["P"] != 16 or row["D"] != 32 for row in gains["joints"].values()):
        raise RuntimeError("Unexpected servo gains")
    for name, row in gains["joints"].items():
        if row["homing_offset"] != calibration.records[name]["homing_offset"]:
            raise RuntimeError("Live homing offset differs from the calibration")
    record = {
        "schema": 1,
        "synthetic": False,
        "status": "incomplete",
        "partition": partition,
        "session_id": output.stem,
        "joint_order": list(MOTOR_NAMES),
        "samples": [],
        "commands": [],
        "amplitude_degrees": amplitude,
        "period_s": period,
        "calibration_sha256": calibration.fingerprint,
        "command_speed_ticks_per_s": speed,
        "alignment_sha256": digest(ROOT / "calibration/kinematics_alignment.json"),
        "pid_snapshot": gains,
        "pid_snapshot_sha256": digest(pid_snapshot),
        "current_is_calibrated_torque": False,
        "versions": {n: importlib.metadata.version(n) for n in ("mujoco", "frax", "numpy")},
    }
    try:
        record["api_arm_config"] = api.call("")
        initial = api.call("/state?fresh=true")
        center = [initial["joints"][n]["position"] for n in MOTOR_NAMES]
        record["last_checked_state"] = initial
        check_state(initial, center, deviation_ticks)
        if any(initial["joints"][n]["moving"] for n in MOTOR_NAMES):
            raise RuntimeError("Arm must be stationary before capture")
        if any(
            c - delta < lo + 20 or c + delta > hi - 20
            for c, lo, hi in zip(center, calibration.tick_min, calibration.tick_max)
        ):
            raise RuntimeError("Excitation is too close to a calibrated joint limit")
        record["center_ticks"] = center
        start = time.monotonic()
        previous_command = -1.0
        last_targets = center[:]
        duration = 2 + 5 * period + 3
        while True:
            elapsed = time.monotonic() - start
            if elapsed > duration:
                break
            if elapsed - previous_command >= 0.095:
                target = center[:]
                phase = elapsed - 2
                if 0 <= phase < 5 * period:
                    joint = min(int(phase / period), 4)
                    target[joint] += round(
                        delta * math.sin(2 * math.pi * (phase % period) / period)
                    )
                sent = time.monotonic() - start
                body = {
                    "joints": {n: {"position": int(t)} for n, t in zip(MOTOR_NAMES, target)},
                    "acc": 10,
                    "wait": False,
                }
                if speed is not None:
                    body["speed"] = speed
                response = api.call("/positions", body)
                received = time.monotonic() - start
                if any(t["clamped"] for t in response["targets"].values()):
                    raise RuntimeError("Unexpected target clamp")
                record["commands"].append(
                    {
                        "t": (sent + received) / 2,
                        "ticks": target,
                        "request_s": received - sent,
                        "accepted_targets": response["targets"],
                    }
                )
                previous_command = elapsed
                last_targets = target
            sent = time.monotonic() - start
            state = api.call("/state?fresh=true")
            received = time.monotonic() - start
            if received - sent > 0.2:
                raise RuntimeError("Excessive telemetry latency")
            record["last_checked_state"] = state
            check_state(state, center, deviation_ticks)
            for name, target in zip(MOTOR_NAMES, last_targets):
                if state["joints"][name]["goal_position"] != target:
                    raise RuntimeError("Goal readback mismatch")
            record["samples"].append(
                {
                    "t": (sent + received) / 2,
                    "ticks": [state["joints"][n]["position"] for n in MOTOR_NAMES],
                    "joints": state["joints"],
                    "request_s": received - sent,
                    "server_timestamp": state["timestamp"],
                }
            )
            if len(record["samples"]) % 20 == 0:
                dump(output, record)
            time.sleep(max(0, 0.05 - (time.monotonic() - start - elapsed)))
        final = np.array(record["samples"][-1]["ticks"])
        if np.any(np.abs(final - center) > 10):
            raise RuntimeError("Capture ended outside the return-to-center tolerance")
        record["status"] = "complete"
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        try:
            record["hold"] = api.call("/stop", {"mode": "hold"})
        except Exception as error:
            record["hold_error"] = repr(error)
            record["status"] = "incomplete"
        dump(output, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid-snapshot", type=Path, required=True)
    parser.add_argument("--partition", choices=("fit", "val", "test"), required=True)
    parser.add_argument("--amplitude", type=float, default=2)
    parser.add_argument("--period", type=float, default=6)
    parser.add_argument("--speed", type=int, help="Omit to use armd's configured nonzero defaults")
    args = parser.parse_args()
    capture(
        args.output,
        args.partition,
        args.pid_snapshot,
        args.amplitude,
        args.period,
        args.speed,
    )
