"""Hardware-safe armd v0.1.0 bridge and encoder-tick <-> Frax radian converter.

Bridges calibrated Frax/MuJoCo joint coordinates (q_enc_rad in R^5, gripper_pct in [0, 1])
to armd (http://127.0.0.1:8765) while enforcing all safety invariants from arm-api.md
and ARCHITECTURE.md:
- 4096 ticks/turn radian conversion via ServoCalibration('follower') without double-applying
  stored EEPROM homing_offset.
- Explicit 4095/4096 scale bridge to LeRobot 0.7 DEGREES.
- Gripper raw tick floor clamped strictly >= 1339 (measured closed jaw stop) to prevent
  STS3215 overload/current faults (HTTP 409 joint_fault).
- Speed strictly >= 1 ticks/s (armd rejects speed == 0 with HTTP 400).
- Partial-dropout frame skip and immediate POST /arms/follower/stop {'mode': 'hold'} on fault.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from brickcell.block_inspector.geometry import REPO_ROOT

TOOLS_DIR = REPO_ROOT / "tools"
ARM_API_DIR = REPO_ROOT / "arm-api"
for _p in (TOOLS_DIR, ARM_API_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from so101_calibration import MOTOR_NAMES, ServoCalibration  # noqa: E402

SAFE_GRIPPER_MIN_RAW = 1339
SAFE_GRIPPER_MAX_RAW = 2829
DEFAULT_STREAM_SPEED_TICKS_S = 420
DEFAULT_STREAM_ACC = 40


class ArmFaultError(RuntimeError):
    """Raised when armd reports an unhealthy servo or blocking joint fault."""


@dataclass(frozen=True)
class DecodedArmState:
    """Validated 6-servo state snapshot decoded into Frax/MuJoCo encoder radians."""

    q_enc_rad: np.ndarray  # (5,) float64
    q_lerobot_deg: np.ndarray  # (5,) float64
    arm_ticks: np.ndarray  # (5,) int64
    gripper_ticks: int
    gripper_pct: float  # [0.0, 1.0] within safe [1339, 2829] operating span
    moving: bool
    healthy: bool
    timestamp: float | None = None


class ArmdFollowerBridge:
    """Safe client adapter between Frax/CogMan/XPolicyLab and armd follower endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        calibration: ServoCalibration | None = None,
        request_fn: Callable[[str, str, dict | None], dict] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.calibration = calibration or ServoCalibration("follower")
        self._request_fn = request_fn
        self._arm_client = None
        if request_fn is None:
            from armd.client import ArmClient

            self._arm_client = ArmClient(base_url=self.base_url, arm="follower")

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        if self._request_fn is not None:
            return self._request_fn(method, path, body)
        assert self._arm_client is not None
        return self._arm_client._request(method, path, body=body)

    # --- Coordinate conversions ----------------------------------------------

    def q_to_arm_ticks(self, q_enc_rad: np.ndarray, clamp: bool = True) -> np.ndarray:
        """Convert 5-DOF Frax/MuJoCo encoder radians to homed STS3215 ticks."""
        q = np.asarray(q_enc_rad, dtype=np.float64)
        if q.shape != (5,) or not np.all(np.isfinite(q)):
            raise ValueError("q_enc_rad must be finite shape (5,)")
        if clamp:
            q = np.clip(q, self.calibration.lower, self.calibration.upper)
        ticks = self.calibration.radians_to_ticks(q)
        if clamp:
            ticks = np.clip(
                ticks,
                self.calibration.tick_min.astype(int),
                self.calibration.tick_max.astype(int),
            )
        return ticks.astype(int)

    def gripper_pct_to_safe_ticks(self, gripper_pct: float) -> int:
        """Map normalized gripper opening [0.0, 1.0] to safe [1339, 2829] raw ticks."""
        val = float(gripper_pct)
        if not np.isfinite(val):
            raise ValueError("gripper_pct must be finite")
        p = float(np.clip(val, 0.0, 1.0))
        ticks = int(round(SAFE_GRIPPER_MIN_RAW + p * (SAFE_GRIPPER_MAX_RAW - SAFE_GRIPPER_MIN_RAW)))
        return max(SAFE_GRIPPER_MIN_RAW, min(SAFE_GRIPPER_MAX_RAW, ticks))

    def safe_ticks_to_gripper_pct(self, raw_ticks: int | float) -> float:
        p = (float(raw_ticks) - SAFE_GRIPPER_MIN_RAW) / float(
            SAFE_GRIPPER_MAX_RAW - SAFE_GRIPPER_MIN_RAW
        )
        return float(np.clip(p, 0.0, 1.0))

    def q_to_lerobot_action6(self, q_enc_rad: np.ndarray, gripper_pct: float) -> np.ndarray:
        """Convert (q_enc_rad, gripper_pct) into LeRobot 0.7 6D action vector (5 deg + 0..100%)."""
        deg5 = self.calibration.radians_to_lerobot_degrees(q_enc_rad)
        grip100 = float(np.clip(gripper_pct, 0.0, 1.0)) * 100.0
        return np.concatenate([deg5, [grip100]]).astype(np.float64)

    def lerobot_action6_to_q(self, action6: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert LeRobot 0.7 6D action vector (5 deg + 0..100%) into (q_enc_rad, gripper_pct)."""
        vec = np.asarray(action6, dtype=np.float64)
        if vec.shape != (6,) or not np.all(np.isfinite(vec)):
            raise ValueError("action6 must be finite shape (6,)")
        q_rad = self.calibration.lerobot_degrees_to_radians(vec[:5])
        q_rad = np.clip(q_rad, self.calibration.lower, self.calibration.upper)
        gripper_pct = float(np.clip(vec[5] / 100.0, 0.0, 1.0))
        return q_rad, gripper_pct

    # --- Payload builders & decoders -----------------------------------------

    def build_positions_payload(
        self,
        q_enc_rad: np.ndarray,
        gripper_pct: float,
        speed: int = DEFAULT_STREAM_SPEED_TICKS_S,
        acc: int = DEFAULT_STREAM_ACC,
        wait: bool = False,
        timeout: float = 5.0,
        tolerance: int = 10,
        include_gripper: bool = True,
    ) -> dict[str, Any]:
        """Build validated POST /arms/follower/positions JSON body."""
        safe_speed = max(1, int(speed))  # armd rejects speed == 0 (servo max speed hazard)
        safe_acc = int(np.clip(int(acc), 0, 254))
        arm_ticks = self.q_to_arm_ticks(q_enc_rad, clamp=True)
        grip_ticks = self.gripper_pct_to_safe_ticks(gripper_pct)

        joints_spec: dict[str, dict[str, int]] = {}
        for name, tick in zip(MOTOR_NAMES, arm_ticks):
            joints_spec[name] = {"position": int(tick), "speed": safe_speed, "acc": safe_acc}
        if include_gripper:
            joints_spec["gripper"] = {
                "position": int(grip_ticks),
                "speed": min(safe_speed, 260),
                "acc": safe_acc,
            }
        return {
            "joints": joints_spec,
            "wait": bool(wait),
            "timeout": float(timeout),
            "tolerance": int(tolerance),
        }

    def decode_state_message(
        self,
        msg: dict[str, Any],
        stop_on_fault: bool = True,
        allow_gripper_overload: bool = False,
    ) -> DecodedArmState | None:
        """Decode GET /arms/follower/state or WS /arms/follower/stream payload.

        Returns None on partial single-servo dropout so 30 Hz consumers skip the frame
        cleanly; triggers emergency hold stop and raises ArmFaultError on blocking fault.
        When `allow_gripper_overload=True` (actively clamping a rigid brick), steady
        `overload`/`current` on `gripper` is expected holding torque and must NOT trigger
        `emergency_stop_hold()` (which would reset gripper goal to present and drop the brick).
        """
        joints_raw = msg.get("joints")
        if isinstance(joints_raw, list):
            joints_by_name = {j["name"]: j for j in joints_raw if "name" in j}
        elif isinstance(joints_raw, dict):
            joints_by_name = joints_raw
        else:
            return None

        required = (*MOTOR_NAMES, "gripper")
        for jname in required:
            j = joints_by_name.get(jname)
            if j is None:
                return None
            if j.get("healthy") is False:
                errs = list(j.get("errors") or [])
                if (
                    allow_gripper_overload
                    and jname == "gripper"
                    and errs
                    and set(errs).issubset({"overload", "current"})
                ):
                    pass
                else:
                    if stop_on_fault:
                        self.emergency_stop_hold()
                    raise ArmFaultError(f"Blocking servo fault on {jname}: errors={errs}")
            if j.get("online") is False or j.get("position") is None:
                return None

        arm_ticks = np.array(
            [int(joints_by_name[n]["position"]) for n in MOTOR_NAMES], dtype=np.int64
        )
        grip_ticks = int(joints_by_name["gripper"]["position"])
        q_rad = self.calibration.ticks_to_radians(arm_ticks)
        q_deg = self.calibration.radians_to_lerobot_degrees(q_rad)
        moving = any(bool(joints_by_name[n].get("moving", False)) for n in required)

        return DecodedArmState(
            q_enc_rad=q_rad,
            q_lerobot_deg=q_deg,
            arm_ticks=arm_ticks,
            gripper_ticks=grip_ticks,
            gripper_pct=self.safe_ticks_to_gripper_pct(grip_ticks),
            moving=moving,
            healthy=True,
            timestamp=msg.get("timestamp") or msg.get("last_poll"),
        )

    # --- Transport operations ------------------------------------------------

    def emergency_stop_hold(self) -> dict:
        """Immediately hold all follower joints at present position (never limp)."""
        return self._call("POST", "/arms/follower/stop", {"mode": "hold"})

    def read_follower_state(self, fresh: bool = False) -> DecodedArmState | None:
        suffix = "?fresh=true" if fresh else "?fresh=false"
        raw = self._call("GET", f"/arms/follower/state{suffix}", None)
        return self.decode_state_message(raw, stop_on_fault=True)

    def send_command(
        self,
        q_enc_rad: np.ndarray,
        gripper_pct: float,
        speed: int = DEFAULT_STREAM_SPEED_TICKS_S,
        acc: int = DEFAULT_STREAM_ACC,
        wait: bool = False,
        timeout: float = 5.0,
    ) -> dict:
        payload = self.build_positions_payload(
            q_enc_rad,
            gripper_pct,
            speed=speed,
            acc=acc,
            wait=wait,
            timeout=timeout,
        )
        try:
            return self._call("POST", "/arms/follower/positions", payload)
        except Exception as exc:
            status = getattr(exc, "status", None)
            code = getattr(exc, "code", "")
            if status == 409 or code == "joint_fault":
                self.emergency_stop_hold()
                raise ArmFaultError(f"armd rejected command with joint_fault: {exc}") from exc
            raise
