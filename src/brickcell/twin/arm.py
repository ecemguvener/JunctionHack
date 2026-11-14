"""Read the follower arm joints (read-only) and convert them to MuJoCo joint angles.

This module imports only the standard library (and scservo_sdk for the servo source). It never moves the arm.

Conversion chain for one arm joint (radians):
  1. Encoder:  q_enc = (raw - mid) * 2*pi/4096, mid = (range_min + range_max) / 2
     (LeRobot calibration, calibration/robots/so_follower/follower.json; raw already has the
     servo homing offset).
  2. CAD:      q_cad = cad_at_zero + cad_sign * q_enc   (calibration/kinematics_alignment.json;
     a human visual match of the real arm to the CAD model).
  3. MuJoCo:   q_mj  = CAD_TO_MUJOCO_SIGN * q_cad + CAD_TO_MUJOCO_OFFSET   (Menagerie SO-101).
     Step 3 is model to model. A forward-kinematics fit of the CAD model
     (models/so101-follower-calibrated.urdf) to the Menagerie model found it. All signs are
     +1; the other 31 sign sets give errors > 70 mm. The CAD link "Body3" and the Menagerie
     body "gripper" have the same frame (the jaw hinge is at the same point), and this fixes
     the wrist_roll offset. Over 200 random poses in the calibration range the TCP
     (site gripperframe) agrees to 1.0 mm mean, 2.6 mm max. tests/test_twin_arm.py repeats
     this check in the MuJoCo env. The wrist_roll value is wrapped to (-pi, pi].

Gripper: percent = (raw - range_min) * 100 / (range_max - range_min); 0 % = closed.
ASSUMPTION: the MuJoCo gripper angle is linear in the percent, from GRIPPER_CLOSED_RAD (0 %) to
GRIPPER_OPEN_RAD (100 %), clipped to the Menagerie joint range. The real jaw angle was not
measured.

Sources (``make_arm_reader``):
  file   JSON file: tools/so101_readonly_snapshot.py output (joints_before.json), a capture
         joints.json, an armd /state response, or {"joints": {name: raw}}.
  armd   armd HTTP API (GET /arms/follower/state). Read endpoints only.
  servo  Direct serial read with SDK READ instructions only (as so101_readonly_snapshot.py).
         It does not write a register and does not change the torque.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from brickcell.twin.state import ARM_JOINTS, ArmState

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_PATH = ROOT / "calibration" / "robots" / "so_follower" / "follower.json"
ALIGNMENT_PATH = ROOT / "calibration" / "kinematics_alignment.json"
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00"
ARMD_URL = "http://127.0.0.1:8765"
SERVO_IDS = {name: i for i, name in enumerate((*ARM_JOINTS, "gripper"), 1)}
RADIANS_PER_TICK = 2 * math.pi / 4096

# q_mj = sign * q_cad + offset (radians). Found by an FK fit (see the module docstring).
# Equivalent encoder form: q_mj = q_enc + ENCODER_TO_MUJOCO_OFFSET (follower calibration).
CAD_TO_MUJOCO_SIGN = {name: 1.0 for name in ARM_JOINTS}
CAD_TO_MUJOCO_OFFSET = {
    "shoulder_pan": 0.36305,
    "shoulder_lift": -1.61170,
    "elbow_flex": 1.53110,
    "wrist_flex": -0.01027,
    "wrist_roll": 3.08435,
}
WRAP_JOINTS = ("wrist_roll",)

# Menagerie SO-101 joint ranges (so101.xml), radians.
MUJOCO_RANGES = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.7453293, 1.7453293),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.658063, 1.658063),
    "wrist_roll": (-2.7438473, 2.8412063),
    "gripper": (-0.174533, 1.7453292),
}
GRIPPER_CLOSED_RAD = 0.0
GRIPPER_OPEN_RAD = 1.7453292


class ArmReadError(RuntimeError):
    """The arm source gave no usable reading."""


# --------------------------------------------------------------------------- conversion


@dataclass(frozen=True)
class ArmCalibration:
    """LeRobot calibration ranges and the CAD alignment of one arm."""

    range_min: dict[str, int]
    range_max: dict[str, int]
    cad_at_zero: dict[str, float]
    cad_sign: dict[str, float]
    sha256: str
    alignment_sha256: str  # calibration hash that the alignment file expects

    @property
    def matches_alignment(self) -> bool:
        return self.sha256 == self.alignment_sha256

    def mid(self, name: str) -> float:
        return (self.range_min[name] + self.range_max[name]) / 2


def load_calibration(
    calibration_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    arm: str = "follower",
) -> ArmCalibration:
    records = json.loads(Path(calibration_path or CALIBRATION_PATH).read_text())
    sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    alignment = json.loads(Path(alignment_path or ALIGNMENT_PATH).read_text())
    profile = alignment["arms"][arm]
    if profile.get("confirmed") is not True:
        raise ValueError(f"{arm}: the CAD alignment is not confirmed")
    return ArmCalibration(
        range_min={n: int(records[n]["range_min"]) for n in SERVO_IDS},
        range_max={n: int(records[n]["range_max"]) for n in SERVO_IDS},
        cad_at_zero={n: float(profile["cad_at_zero_rad"][n]) for n in ARM_JOINTS},
        cad_sign={n: float(profile["cad_sign"][n]) for n in ARM_JOINTS},
        sha256=sha,
        alignment_sha256=str(profile.get("calibration_sha256", "")),
    )


def wrap(angle: float) -> float:
    """Angle in (-pi, pi]."""
    return math.pi - (math.pi - angle) % (2 * math.pi)


def encoder_rad(raw: float, name: str, cal: ArmCalibration) -> float:
    return (float(raw) - cal.mid(name)) * RADIANS_PER_TICK


def cad_rad(raw: float, name: str, cal: ArmCalibration) -> float:
    return cal.cad_at_zero[name] + cal.cad_sign[name] * encoder_rad(raw, name, cal)


def cad_to_mujoco(name: str, q_cad: float) -> float:
    q = CAD_TO_MUJOCO_SIGN[name] * q_cad + CAD_TO_MUJOCO_OFFSET[name]
    return wrap(q) if name in WRAP_JOINTS else q


def gripper_percent(raw: float, cal: ArmCalibration) -> float:
    lo, hi = cal.range_min["gripper"], cal.range_max["gripper"]
    return (float(raw) - lo) * 100.0 / (hi - lo)


def gripper_percent_to_mujoco(
    percent: float, closed: float = GRIPPER_CLOSED_RAD, opened: float = GRIPPER_OPEN_RAD
) -> float:
    q = closed + (opened - closed) * float(percent) / 100.0
    lo, hi = MUJOCO_RANGES["gripper"]
    return min(max(q, lo), hi)


def convert_ticks(
    ticks: dict[str, float],
    cal: ArmCalibration,
    timestamp: float = 0.0,
    source: str = "",
    gripper_closed: float = GRIPPER_CLOSED_RAD,
    gripper_open: float = GRIPPER_OPEN_RAD,
) -> tuple[ArmState, dict]:
    """Raw ticks {joint: raw} -> (ArmState in the MuJoCo convention, details).

    ``details`` has the encoder, CAD and MuJoCo angles per joint, the gripper percent, and the
    joints that are out of the calibration range or of the MuJoCo joint range.
    """
    missing = [n for n in ARM_JOINTS if n not in ticks]
    if missing:
        raise ArmReadError(f"No reading for joints {missing}")
    joints, details = {}, {"joints": {}, "out_of_calibration": [], "out_of_mujoco_range": []}
    for name in ARM_JOINTS:
        raw = float(ticks[name])
        q_enc = encoder_rad(raw, name, cal)
        q_cad = cal.cad_at_zero[name] + cal.cad_sign[name] * q_enc
        q_mj = cad_to_mujoco(name, q_cad)
        joints[name] = q_mj
        details["joints"][name] = {"raw": raw, "q_enc": q_enc, "q_cad": q_cad, "q_mujoco": q_mj}
        if not cal.range_min[name] <= raw <= cal.range_max[name]:
            details["out_of_calibration"].append(name)
        lo, hi = MUJOCO_RANGES[name]
        if not lo <= q_mj <= hi:
            details["out_of_mujoco_range"].append(name)
    gripper = None
    if ticks.get("gripper") is not None:
        pct = gripper_percent(ticks["gripper"], cal)
        gripper = gripper_percent_to_mujoco(pct, gripper_closed, gripper_open)
        details["gripper"] = {"raw": float(ticks["gripper"]), "percent": pct, "q_mujoco": gripper}
    if not cal.matches_alignment:
        details["warning"] = "calibration file hash differs from the alignment file"
    state = ArmState(joints=joints, gripper=gripper, timestamp=timestamp, source=source)
    return state, details


# --------------------------------------------------------------------------- tick parsing


def ticks_from_json(data: dict, arm: str = "follower") -> tuple[dict[str, float], float]:
    """Raw ticks and a unix timestamp from the known JSON layouts (see the module docstring)."""
    if "arms" in data:  # so101_readonly_snapshot.py output or a capture joints.json
        body = data["arms"][arm]
        joints = body["joints"]
        stamp = data.get("timestamp_utc") or data.get("timestamp_before_utc")
    else:  # armd /arms/<arm>/state, or {"joints": {name: raw}}
        joints = data.get("joints", data)
        stamp = data.get("timestamp")
    ticks = {}
    for name in SERVO_IDS:
        value = joints.get(name)
        if isinstance(value, dict):
            value = value.get("raw", value.get("position"))
        if value is not None:
            ticks[name] = float(value)
    return ticks, _unix(stamp)


def _unix(stamp) -> float:
    if stamp is None:
        return 0.0
    if isinstance(stamp, (int, float)):
        return float(stamp)
    from datetime import datetime

    return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()


# --------------------------------------------------------------------------- readers


class ArmReader:
    """Base class. ``read()`` returns (ArmState, details) or raises ArmReadError."""

    source = "none"

    def __init__(self, cal: ArmCalibration | None = None, gripper_closed=None, gripper_open=None):
        self.cal = cal or load_calibration()
        self.gripper_closed = GRIPPER_CLOSED_RAD if gripper_closed is None else gripper_closed
        self.gripper_open = GRIPPER_OPEN_RAD if gripper_open is None else gripper_open

    def ticks(self) -> tuple[dict[str, float], float]:
        raise NotImplementedError

    def read(self) -> tuple[ArmState, dict]:
        ticks, stamp = self.ticks()
        return convert_ticks(
            ticks,
            self.cal,
            stamp or time.time(),
            self.source,
            self.gripper_closed,
            self.gripper_open,
        )

    def close(self) -> None:
        pass


class FileArmReader(ArmReader):
    """Read ticks from a JSON file. The file is read again at each call."""

    source = "file"

    def __init__(self, path: str | Path, **kwargs):
        super().__init__(**kwargs)
        self.path = Path(path)

    def ticks(self):
        try:
            return ticks_from_json(json.loads(self.path.read_text()))
        except (OSError, ValueError, KeyError) as exc:
            raise ArmReadError(f"arm file {self.path}: {exc}") from None


class ArmdArmReader(ArmReader):
    """Read ticks from armd (GET /arms/<arm>/state). It uses only the read endpoint."""

    source = "armd"

    def __init__(self, url: str = ARMD_URL, arm: str = "follower", timeout: float = 2.0, **kw):
        super().__init__(**kw)
        self.url, self.arm, self.timeout = url.rstrip("/"), arm, timeout

    def ticks(self):
        url = f"{self.url}/arms/{self.arm}/state?fresh=false"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ArmReadError(f"armd {url}: {exc}") from None
        if data.get("timestamp") is None:
            raise ArmReadError("armd has no reading yet")
        return ticks_from_json(data)


class ServoArmReader(ArmReader):
    """Direct serial read of the follower servos. SDK READ instructions only.

    It refuses a port that another process holds (armd, LeRobot). It reads Present_Position
    (address 56) and Torque_Enable (address 40). It never writes a register.
    NOT TESTED LIVE: the user has not approved live arm reads yet.
    """

    source = "servo-readonly"
    PRESENT_POSITION = 56

    def __init__(self, port: str = FOLLOWER_PORT, **kwargs):
        super().__init__(**kwargs)
        self.port_name = port

    def ticks(self):
        try:
            from scservo_sdk import COMM_SUCCESS, PortHandler, sms_sts
        except ImportError as exc:
            raise ArmReadError(f"servo source needs scservo_sdk (feetech-servo-sdk): {exc}")
        owner = subprocess.run(["fuser", self.port_name], capture_output=True)
        if owner.returncode != 1:
            raise ArmReadError(f"{self.port_name} is in use by another process (or no fuser)")
        port = PortHandler(self.port_name)
        try:
            if not port.openPort() or not port.setBaudRate(1_000_000):
                raise ArmReadError(f"cannot open {self.port_name}")
            sdk = sms_sts(port)
            ticks = {}
            for name, sid in SERVO_IDS.items():
                raw, status, error = sdk.read2ByteTxRx(sid, self.PRESENT_POSITION)
                if status != COMM_SUCCESS or error or not 0 <= raw <= 4095:
                    raise ArmReadError(f"{name}: bad position read {raw}, {status}, {error}")
                ticks[name] = float(raw)
            return ticks, time.time()
        finally:
            port.closePort()


def make_arm_reader(source: str, path: str | Path | None = None, config: dict | None = None):
    """``source``: none, file, armd or servo. Returns None for "none"."""
    config = config or {}
    kwargs = {
        "gripper_closed": config.get("gripper_closed_rad"),
        "gripper_open": config.get("gripper_open_rad"),
    }
    if source in (None, "", "none"):
        return None
    cal = load_calibration(config.get("calibration"), config.get("alignment"))
    if source == "file":
        if not path:
            raise ValueError("--arm-source file needs --arm-file PATH")
        return FileArmReader(path, cal=cal, **kwargs)
    if source == "armd":
        return ArmdArmReader(config.get("armd_url", ARMD_URL), cal=cal, **kwargs)
    if source == "servo":
        return ServoArmReader(config.get("servo_port", FOLLOWER_PORT), cal=cal, **kwargs)
    raise ValueError(f"Unknown arm source {source!r}; use none, file, armd or servo")


def main() -> None:
    """Convert one JSON file and print the MuJoCo joint angles (offline tool)."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert recorded arm ticks to MuJoCo angles.")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    state, details = FileArmReader(args.file).read()
    print(json.dumps({"arm": state.__dict__, "details": details}, indent=1))


if __name__ == "__main__":
    main()
