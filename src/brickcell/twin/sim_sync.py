"""Apply a measured StationState to the MuJoCo World, so that the sim copies the real station.

MuJoCo environment only. Frames and units: see brickcell.twin.state.

Steps of ``apply_state``:
  1. Plates: move the static plate bodies (model.body_pos). Collision follows after mj_forward.
     A plate that is not in the state keeps its place. Exception: a plate in
     ``park_absent_plates`` (default: the spare plate) goes to its park spot.
  2. Bricks: set each freejoint from the measured mesh pose (all copies, BRICK_IDS). A brick
     that is not in the state is "parked" on the table at a spot that no camera sees. A parked
     brick has no contacts and gravity compensation (World.set_parked).
  3. Arm: set the joint angles, the actuator targets and the World command state. The World
     then holds these joint angles (World.hold_q) until the next move().
  4. Relax: step the physics with zero velocity after each step. This pushes bricks out of
     penetration and lets a brick that floats a little come down, without a bounce.
  5. Settle: step the normal physics for ``settle_s``. Then compare the settled brick poses
     with the measured poses. A large drift is a flag (for example, a pose that is not stable).
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml

from brickcell.simulation.scene_config import color_hex
from brickcell.simulation.world import PARK, PARK_PLATES
from brickcell.twin.state import (
    ARM_JOINTS,
    BRICK_IDS,
    PLATE_NAMES,
    ArmState,
    BrickState,
    PlateState,
    StationState,
    body_pose_from_mesh_pose,
    mesh_pose_from_body_pose,
    part_of,
    quat_to_matrix,
    rotation_angle_deg,
)

ROOT = Path(__file__).resolve().parents[3]
TWIN_CONFIG = ROOT / "configs" / "twin.yaml"
STABLE_POSES = ROOT / "assets" / "duplo" / "stable_poses.json"
MAX_LIVE_HZ = 2.0
PLATE_RADIUS = 0.085  # m; inner radius of the KALAS plate wall at the floor
GRIPPER = "gripper"

DEFAULT_SIM_CONFIG: dict = {
    "station_url": "http://127.0.0.1:8095",  # station service; GET {station_url}/state
    "fetch_timeout_s": 3.0,
    "fresh_timeout_s": 15.0,  # for GET /state?fresh=1 (the station captures new frames)
    "live_hz": 1.0,  # live polling rate; MAX_LIVE_HZ is the limit
    "relax_s": 0.2,  # quasi-static steps: remove penetration, close gaps up to ~4 mm
    "settle_s": 0.5,  # normal physics after the relax step, arm held
    "max_drift_mm": 3.0,  # larger drift from the measured pose is a flag
    "max_drift_deg": 5.0,
    "max_penetration_mm": 5.0,  # deeper measured penetration is a flag
    # Brick colours: copy the observed colour of each measured brick to the sim brick.
    "apply_colors": True,
    # Observed camera colour -> sim albedo. The sim lights make a surface darker: an albedo of
    # 0.68 renders as 0.631 (floor check, 2026-11-14), so the gain is 0.68 / 0.631.
    "color_gain": 1.08,
    # Change the sim colour only if a channel differs by more than this (0..1). A colour change
    # makes the browser viewer rebuild its meshes, so live mode must not do it each frame.
    "min_color_change": 0.06,
    # Park positions (x, y) on the table for bricks that are not in the state, one for each
    # brick body (BRICK_IDS). Default: the world.PARK grid, which no camera sees. The config
    # file can override single entries.
    "park": {oid: list(xy) for oid, xy in PARK.items()},
    # Plates that go to their park spot when the state does not have them. The other plates
    # keep their place.
    "park_absent_plates": ["spare"],
    "park_plates": {name: list(xy) for name, xy in PARK_PLATES.items()},
}


# --------------------------------------------------------------------------- config


def load_sim_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """The ``sim:`` section of configs/twin.yaml, with defaults. ``overrides`` wins over the file."""
    path = Path(path) if path is not None else TWIN_CONFIG
    raw = {}
    if path.exists():
        raw = (yaml.safe_load(path.read_text()) or {}).get("sim") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 'sim' must be a mapping")
    return resolve_sim_config({**raw, **(overrides or {})})


def resolve_sim_config(config: dict | None) -> dict:
    """Merge ``config`` into the defaults and check the values."""
    config = dict(config or {})
    unknown = set(config) - set(DEFAULT_SIM_CONFIG)
    if unknown:
        raise ValueError(f"Unknown sim config keys {sorted(unknown)}")
    park = {**DEFAULT_SIM_CONFIG["park"], **(config.pop("park", None) or {})}
    park_plates = {**DEFAULT_SIM_CONFIG["park_plates"], **(config.pop("park_plates", None) or {})}
    result = {**DEFAULT_SIM_CONFIG, **config, "park": park, "park_plates": park_plates}
    for names, entries in ((BRICK_IDS, park), (PLATE_NAMES, park_plates)):
        for name, xy in entries.items():
            if name not in names or len(xy) not in (2, 3):
                raise ValueError(f"Bad park entry {name!r}: {xy!r}")
            entries[name] = [float(v) for v in xy[:2]]
    absent = result["park_absent_plates"]
    if isinstance(absent, str) or not all(name in park_plates for name in absent or []):
        raise ValueError(f"sim.park_absent_plates: names with a park_plates entry, not {absent!r}")
    result["park_absent_plates"] = list(absent or [])
    result["apply_colors"] = bool(result["apply_colors"])
    for key in ("color_gain", "min_color_change"):
        result[key] = float(result[key])
    if result["color_gain"] <= 0:
        raise ValueError("sim.color_gain must be > 0")
    for key in ("fetch_timeout_s", "fresh_timeout_s", "live_hz", "relax_s", "settle_s"):
        result[key] = float(result[key])
        if result[key] < 0:
            raise ValueError(f"sim.{key} must be >= 0")
    if not 0 < result["live_hz"]:
        raise ValueError("sim.live_hz must be > 0")
    result["live_hz"] = min(result["live_hz"], MAX_LIVE_HZ)
    result["station_url"] = str(result["station_url"]).rstrip("/")
    return result


# --------------------------------------------------------------------------- helpers


def _freejoint(model, object_id: str) -> tuple[int, int]:
    joint = model.joint(f"{object_id}-joint")
    return int(joint.qposadr[0]), int(joint.dofadr[0])


def _gripper_adr(model) -> tuple[int, int]:
    joint = model.joint(GRIPPER)
    return int(joint.qposadr[0]), int(joint.dofadr[0])


def _arm_dofs(world) -> list[int]:
    return [*world.ik.dof_adr.tolist(), _gripper_adr(world.model)[1]]


def _body_pose(world, object_id: str) -> tuple[np.ndarray, np.ndarray]:
    a, _ = _freejoint(world.model, object_id)
    return world.data.qpos[a : a + 3].copy(), world.data.qpos[a + 3 : a + 7].copy()


def _mesh_pose(world, object_id: str) -> tuple[np.ndarray, np.ndarray]:
    return mesh_pose_from_body_pose(object_id, *_body_pose(world, object_id))


def _set_body_pose(world, object_id: str, pos, quat) -> None:
    a, v = _freejoint(world.model, object_id)
    world.data.qpos[a : a + 3] = pos
    world.data.qpos[a + 3 : a + 7] = np.asarray(quat, float) / np.linalg.norm(quat)
    world.data.qvel[v : v + 6] = 0


def _park(world, object_id: str, config: dict) -> None:
    world.park(object_id, config["park"][object_id])


def is_parked(world, object_id: str, config: dict | None = None) -> bool:
    """True if the brick is parked (no contacts, see World.set_parked). ``config`` is not used."""
    return bool(world.is_parked(object_id))


def plate_is_parked(world, name: str, config: dict) -> bool:
    """True if the plate stands near its park spot (only plates with a park_plates entry)."""
    if name not in config["park_plates"]:
        return False
    pos = world.model.body_pos[world.model.body(f"plate_{name}").id]
    return float(np.linalg.norm(pos[:2] - config["park_plates"][name])) < 0.05


def _penetration_mm(world, object_ids) -> dict[str, float]:
    """Deepest contact penetration of each brick solid (mm, >= 0). Needs mj_forward first."""
    m, d = world.model, world.data
    geoms = {m.geom(f"{oid}-solid").id: oid for oid in object_ids}
    depth = dict.fromkeys(object_ids, 0.0)
    for i in range(d.ncon):
        contact = d.contact[i]
        for g in (contact.geom1, contact.geom2):
            if g in geoms and contact.dist < 0:
                depth[geoms[g]] = max(depth[geoms[g]], -1000 * float(contact.dist))
    return depth


def _support(world, pos) -> str:
    """Name of the support below a mesh origin: a plate, the table or "other"."""
    for name in PLATE_NAMES:
        centre = world.data.xpos[world.model.body(f"plate_{name}").id]
        if np.linalg.norm(np.asarray(pos[:2]) - centre[:2]) < PLATE_RADIUS and pos[2] < 0.03:
            return f"plate_{name}"
    return "table" if pos[2] < 0.03 else "other"


@lru_cache(maxsize=1)
def _stable_poses() -> dict:
    return json.loads(STABLE_POSES.read_text())["bricks"]


def match_stable_pose(object_id: str, quat, max_deg: float = 5.0) -> str | None:
    """Id of the stable pose that equals ``quat`` (mesh frame) up to a yaw about world Z."""
    r = quat_to_matrix(quat)
    best, best_angle = None, max_deg
    for pose in _stable_poses().get(part_of(object_id).object_id, {}).get("poses", []):
        rel = r @ quat_to_matrix(pose["quat"]).T  # a pure yaw if the tilt is the same
        angle = math.degrees(math.acos(max(-1.0, min(1.0, rel[2, 2]))))
        if angle < best_angle:
            best, best_angle = pose["id"], angle
    return best


def _hold_arm(world, q: np.ndarray, grip: float) -> None:
    """Set the arm joints, actuator targets and World command state to ``q`` and ``grip``."""
    m, d = world.model, world.data
    d.qpos[world.ik.qpos_adr] = q
    d.qvel[world.ik.dof_adr] = 0
    ga, gv = _gripper_adr(m)
    d.qpos[ga], d.qvel[gv] = grip, 0
    ctrl = np.array([*q, grip])
    ranges = m.actuator_ctrlrange[world.actuators]
    limited = m.actuator_ctrllimited[world.actuators].astype(bool)
    ctrl[limited] = np.clip(ctrl[limited], ranges[limited, 0], ranges[limited, 1])
    d.ctrl[world.actuators] = ctrl
    world.q = np.asarray(q, float).copy()
    world.hold_q = world.q.copy()
    world.grip = float(grip)


def _apply_arm(world, arm: ArmState) -> dict:
    m = world.model
    q = world.data.qpos[world.ik.qpos_adr].copy()
    clamps, missing = {}, []
    for i, name in enumerate(ARM_JOINTS):
        if name not in arm.joints:
            missing.append(name)
            continue
        low, high = m.jnt_range[m.joint(name).id]
        value = float(arm.joints[name])
        q[i] = min(max(value, low), high)
        if q[i] != value:
            clamps[name] = {"given": value, "used": float(q[i])}
    grip = world.grip
    if arm.gripper is not None:
        low, high = m.jnt_range[m.joint(GRIPPER).id]
        grip = min(max(float(arm.gripper), low), high)
        if grip != arm.gripper:
            clamps[GRIPPER] = {"given": float(arm.gripper), "used": float(grip)}
    _hold_arm(world, q, grip)
    return {
        "joints": dict(zip(ARM_JOINTS, q.tolist())),
        "gripper": float(grip),
        "clamped": clamps,
        "missing": missing,
        "source": arm.source,
    }


def _step(world, seconds: float, quasi_static: bool, dofs: list[int]) -> None:
    """Step the physics with the arm held. ``quasi_static``: zero ``dofs`` velocities each step."""
    m, d = world.model, world.data
    steps = round(seconds / m.opt.timestep)
    if not quasi_static:
        if steps > 0:
            mujoco.mj_step(m, d, nstep=steps)
        return
    for _ in range(steps):
        mujoco.mj_step(m, d)
        d.qvel[dofs] = 0


# --------------------------------------------------------------------------- public API


def apply_state(world, state: StationState, config: dict | None = None) -> dict:
    """Make ``world`` copy ``state``. Main thread of the World only. Returns a JSON report.

    ``config``: sim config (see DEFAULT_SIM_CONFIG); None loads configs/twin.yaml. Use
    ``{"relax_s": 0, "settle_s": 0}`` to put the bricks exactly at the measured poses.
    """
    config = load_sim_config() if config is None else resolve_sim_config(config)
    started = time.perf_counter()
    m, d = world.model, world.data
    report: dict = {
        "timestamp": state.timestamp,
        "applied_at": time.time(),
        "bricks": {},
        "hidden": [],
        "plates": {},
        "arm": None,
        "colors_changed": {},
        "warnings": [],
        "flags": [],
    }

    # 1. Plates.
    given = {plate.name: plate for plate in state.plates}
    if len(given) != len(state.plates):
        raise ValueError("A plate is in the state two times")
    for name in PLATE_NAMES:
        body = m.body(f"plate_{name}").id
        old = m.body_pos[body].copy()
        if name in given:
            pos = given[name].pos
            entry = {"given_z_mm": round(1000 * float(pos[2]), 2), "parked": False}
        elif name in config["park_absent_plates"]:
            pos = config["park_plates"][name]
            entry = {"parked": True}
        else:
            continue  # not measured: the plate keeps its place
        m.body_pos[body] = [pos[0], pos[1], 0.0]  # a plate stands on the table
        if name in world.bowls:  # sort targets only
            world.bowls[name][:2] = pos[:2]
        report["plates"][name] = {
            "pos": [float(pos[0]), float(pos[1]), 0.0],
            "moved_mm": round(1000 * float(np.linalg.norm(m.body_pos[body] - old)), 2),
            **entry,
        }

    # 2. Bricks.
    measured: dict[str, BrickState] = {}
    for brick in state.bricks:
        if brick.object_id in measured:
            raise ValueError(f"Brick {brick.object_id} is in the state two times")
        sim_part = part_of(brick.object_id).part
        if brick.part != sim_part:
            report["warnings"].append(
                f"{brick.object_id}: part {brick.part} given, the sim has {sim_part}"
            )
        measured[brick.object_id] = brick
    for object_id in BRICK_IDS:
        if object_id in measured:
            b = measured[object_id]
            world.set_parked(object_id, False)
            _set_body_pose(world, object_id, *body_pose_from_mesh_pose(object_id, b.pos, b.quat))
            if config["apply_colors"] and len(b.color) == 3:
                rgb = np.clip(np.asarray(b.color, float) * config["color_gain"], 0.0, 1.0)
                old = np.asarray(world.bricks[object_id].rgba, float)
                if np.abs(rgb - old).max() > config["min_color_change"]:
                    world.set_brick_color(object_id, [float(v) for v in rgb])
                    report["colors_changed"][object_id] = color_hex(rgb)
        else:
            _park(world, object_id, config)
            report["hidden"].append(object_id)

    # 3. Arm.
    if state.arm is not None:
        report["arm"] = _apply_arm(world, state.arm)
        for name, clamp in report["arm"]["clamped"].items():
            report["flags"].append(f"arm joint {name} clamped to its range: {clamp}")
    else:
        # Hold the arm where it is, so that the settle does not move it.
        _hold_arm(world, d.qpos[world.ik.qpos_adr].copy(), world.grip)

    mujoco.mj_forward(m, d)
    penetration = _penetration_mm(world, measured)
    after_set = {oid: _mesh_pose(world, oid) for oid in measured}

    # 4 and 5. Relax, then settle. Velocities of the bricks and the arm stay zero in the relax.
    dofs = [_freejoint(m, oid)[1] + k for oid in BRICK_IDS for k in range(6)] + _arm_dofs(world)
    _step(world, config["relax_s"], quasi_static=True, dofs=dofs)
    relaxed = {oid: _mesh_pose(world, oid) for oid in measured}
    _step(world, config["settle_s"], quasi_static=False, dofs=dofs)
    mujoco.mj_forward(m, d)
    world.target = world.ik.tcp(d).copy()

    for object_id, brick in measured.items():
        pos, quat = _mesh_pose(world, object_id)
        _, v = _freejoint(m, object_id)
        drift_mm = 1000 * float(np.linalg.norm(pos - np.asarray(brick.pos, float)))
        drift_deg = rotation_angle_deg(quat_to_matrix(brick.quat), quat_to_matrix(quat))
        entry = {
            "measured": {"pos": list(map(float, brick.pos)), "quat": list(map(float, brick.quat))},
            "pos": pos.tolist(),
            "quat": quat.tolist(),
            "penetration_mm": round(penetration[object_id], 3),
            "relax_dz_mm": round(
                1000 * float(relaxed[object_id][0][2] - after_set[object_id][0][2]), 3
            ),
            "drift_mm": round(drift_mm, 3),
            "drift_xy_mm": round(1000 * float(np.linalg.norm(pos[:2] - brick.pos[:2])), 3),
            "drift_z_mm": round(1000 * float(pos[2] - brick.pos[2]), 3),
            "drift_deg": round(drift_deg, 3),
            # Movement in the settle only (after the relax): a large value = not a stable pose.
            "settle_drift_mm": round(1000 * float(np.linalg.norm(pos - relaxed[object_id][0])), 3),
            "settle_drift_deg": round(
                rotation_angle_deg(quat_to_matrix(relaxed[object_id][1]), quat_to_matrix(quat)), 3
            ),
            "speed_mm_s": round(1000 * float(np.linalg.norm(d.qvel[v : v + 3])), 3),
            "support": _support(world, pos),
            "measured_support": brick.support,
            "stable_pose": match_stable_pose(object_id, quat),
            "flags": [],
        }
        if drift_mm > config["max_drift_mm"]:
            entry["flags"].append(f"drift {drift_mm:.1f} mm > {config['max_drift_mm']} mm")
        if drift_deg > config["max_drift_deg"]:
            entry["flags"].append(f"rotation {drift_deg:.1f} deg > {config['max_drift_deg']} deg")
        if penetration[object_id] > config["max_penetration_mm"]:
            entry["flags"].append(f"measured penetration {penetration[object_id]:.1f} mm")
        report["flags"] += [f"{object_id}: {flag}" for flag in entry["flags"]]
        report["bricks"][object_id] = entry

    if state.arm is not None:
        q = d.qpos[world.ik.qpos_adr]
        report["arm"]["hold_error_deg"] = round(
            math.degrees(float(np.max(np.abs(q - world.hold_q)))), 3
        )
    report["ok"] = not report["flags"]
    report["relax_s"], report["settle_s"] = config["relax_s"], config["settle_s"]
    report["compute_ms"] = round(1000 * (time.perf_counter() - started), 1)
    return report


def state_from_world(
    world, timestamp: float | None = None, config: dict | None = None
) -> StationState:
    """The current sim bricks, plates and arm as a StationState.

    Parked bricks and parked plates (see ``plate_is_parked``) are left out.
    """
    config = resolve_sim_config(config)
    m, d = world.model, world.data
    mujoco.mj_kinematics(m, d)
    bricks = []
    for object_id in BRICK_IDS:
        if is_parked(world, object_id, config):
            continue
        pos, quat = _mesh_pose(world, object_id)
        bricks.append(
            BrickState(
                object_id=object_id,
                part=part_of(object_id).part,
                pos=pos.tolist(),
                quat=quat.tolist(),
                color=[float(c) for c in world.bricks[object_id].rgba],
                stable_pose=match_stable_pose(object_id, quat),
                support=_support(world, pos),
                confidence=1.0,
                residual_mm=0.0,
                cameras=[],
            )
        )
    plates = [
        PlateState(name, m.body_pos[m.body(f"plate_{name}").id].tolist(), 1.0)
        for name in PLATE_NAMES
        if not plate_is_parked(world, name, config)
    ]
    ga, _ = _gripper_adr(m)
    arm = ArmState(
        joints=dict(zip(ARM_JOINTS, d.qpos[world.ik.qpos_adr].tolist())),
        gripper=float(d.qpos[ga]),
        timestamp=time.time() if timestamp is None else timestamp,
        source="sim",
    )
    return StationState(
        timestamp=time.time() if timestamp is None else timestamp,
        bricks=bricks,
        plates=plates,
        arm=arm,
        diagnostics={"source": "mujoco"},
    )


# --------------------------------------------------------------------------- server controller


class TwinConflict(RuntimeError):
    """The sim cannot copy the station now (a pick / replay run is active). HTTP 409."""

    status = 409


class TwinSync:
    """Station twin endpoints and live mode for the sim server. No viewer code here.

    ``call_on_main(function)`` runs a function on the World thread and returns its result.
    ``run_active()`` is True while a pick / replay run is active. ``on_applied(report)`` runs on
    the World thread after each apply (for example, to refresh a viewer). ``on_change()`` runs
    when the live flag or the status changes (any thread).
    """

    def __init__(
        self,
        world,
        config: dict | None = None,
        call_on_main=None,
        run_active=lambda: False,
        on_applied=lambda report: None,
        on_change=lambda: None,
    ):
        self.world = world
        self.config = load_sim_config() if config is None else resolve_sim_config(config)
        self.call_on_main = call_on_main or (lambda function: function())
        self.run_active = run_active
        self.on_applied = on_applied
        self.on_change = on_change
        self.lock = threading.Lock()
        self.state: StationState | None = None
        self.report: dict | None = None
        self.error: str | None = None
        self.source: str | None = None
        self.received: float | None = None
        self.live = False
        self.hz = self.config["live_hz"]
        self.applied = 0
        self.skipped = 0
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ actions

    def _set_error(self, error: str | None) -> None:
        with self.lock:
            self.error = error
        self.on_change()

    def apply(self, state: StationState | dict, source: str = "post") -> dict:
        """Validate and apply a state on the World thread. Returns the report."""
        try:
            if not isinstance(state, StationState):
                try:
                    state = StationState.from_dict(state)
                except (TypeError, KeyError, AttributeError) as exc:
                    raise ValueError(f"Bad station state: {exc}") from None
            if self.run_active():
                raise TwinConflict("A pick / replay run is active: the twin does not apply now")

            def main() -> dict:
                if self.run_active():  # the run can start while the request waits
                    raise TwinConflict("A pick / replay run is active: the twin does not apply now")
                report = apply_state(self.world, state, self.config)
                self.on_applied(report)
                return report

            report = self.call_on_main(main)
        except Exception as exc:
            self._set_error(f"{type(exc).__name__}: {exc}")
            raise
        with self.lock:
            self.state, self.report, self.source = state, report, source
            self.received, self.error = time.time(), None
            self.applied += 1
        self.on_change()
        return report

    def fetch(self, fresh: bool = False) -> StationState:
        """GET {station_url}/state once. A body {"state": {...}} is also accepted."""
        url = self.config["station_url"] + "/state" + ("?fresh=1" if fresh else "")
        timeout = self.config["fresh_timeout_s" if fresh else "fetch_timeout_s"]
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - config URL
            raw = json.loads(response.read(4_000_000))
        if isinstance(raw, dict) and "timestamp" not in raw and isinstance(raw.get("state"), dict):
            raw = raw["state"]
        return StationState.from_dict(raw)

    def sync(self, fresh: bool = False, source: str = "sync") -> dict:
        """Fetch the station state once and apply it."""
        try:
            state = self.fetch(fresh)
        except Exception as exc:
            self._set_error(f"fetch {self.config['station_url']}/state: {exc}")
            raise
        return self.apply(state, source)

    def set_live(self, enabled: bool, hz: float | None = None) -> dict:
        """Start or stop live polling of the station (at most MAX_LIVE_HZ)."""
        with self.lock:
            if hz is not None:
                if not float(hz) > 0:
                    raise ValueError("hz must be > 0")
                self.hz = min(float(hz), MAX_LIVE_HZ)
            if enabled and self.run_active():
                raise TwinConflict("A pick / replay run is active: live mode stays off")
            changed = self.live != bool(enabled)
            self.live = bool(enabled)
            if self.live and self._thread is None:
                self._thread = threading.Thread(target=self._live_loop, daemon=True)
                self._thread.start()
        self._wake.set()
        if changed:
            self.on_change()
        return self.status()

    def stop_live(self, reason: str) -> None:
        """Stop live mode (for example, a run starts)."""
        with self.lock:
            if not self.live:
                return
            self.live = False
            self.error = f"Live mode stopped: {reason}"
        self._wake.set()
        self.on_change()

    def _live_loop(self) -> None:
        last_timestamp = None
        while True:
            with self.lock:
                live, period = self.live, 1.0 / self.hz
            if not live:
                last_timestamp = None
                self._wake.wait()
                self._wake.clear()
                continue
            started = time.monotonic()
            if self.run_active():
                self.stop_live("a pick / replay run started")
                continue
            try:
                state = self.fetch()
                if state.timestamp == last_timestamp:
                    with self.lock:
                        self.skipped += 1
                else:
                    self.apply(state, "live")
                    last_timestamp = state.timestamp
            except TwinConflict:
                self.stop_live("a pick / replay run started")
                continue
            except Exception as exc:  # keep polling: the station can come back
                self._set_error(f"live: {type(exc).__name__}: {exc}")
            self._wake.wait(max(0.0, period - (time.monotonic() - started)))
            self._wake.clear()

    # ------------------------------------------------------------------ status

    def status(self) -> dict:
        with self.lock:
            now = time.time()
            state, report = self.state, self.report
            return {
                "live": self.live,
                "hz": self.hz,
                "station_url": self.config["station_url"],
                "run_active": bool(self.run_active()),
                "source": self.source,
                "state": state.to_dict() if state else None,
                "state_age_s": round(now - state.timestamp, 3) if state else None,
                "received_age_s": round(now - self.received, 3) if self.received else None,
                "bricks": len(state.bricks) if state else 0,
                "report": report,
                "error": self.error,
                "applied": self.applied,
                "skipped_same_timestamp": self.skipped,
            }

    def status_text(self) -> str:
        """Short Markdown status for a viewer panel."""
        s = self.status()
        lines = [
            f"Live: **{'on' if s['live'] else 'off'}** ({s['hz']:g} Hz) · `{s['station_url']}`"
        ]
        if s["state"] is None:
            lines.append("No station state applied yet.")
        else:
            report = s["report"] or {}
            lines.append(
                f"State age {s['state_age_s']:.1f} s · {s['bricks']} bricks · "
                f"{len(report.get('hidden', []))} parked · from {s['source']}"
            )
            if report.get("flags"):
                lines.append("Flags: " + "; ".join(report["flags"][:3]))
        if s["error"]:
            lines.append(f"**Error:** {s['error']}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ HTTP

    def request(self, method: str, path: str, payload: dict) -> dict:
        """Serve /twin/state, /twin/sync, /twin/live and /twin/status."""
        path = path.split("?", 1)[0]
        if method == "POST" and path == "/twin/state":
            return self.apply(payload, "post")
        if method == "POST" and path == "/twin/sync":
            return self.sync(fresh=bool((payload or {}).get("fresh", False)))
        if method == "POST" and path == "/twin/live":
            if "enabled" not in payload:
                raise ValueError('Give {"enabled": true|false, "hz": 1.0}')
            return self.set_live(bool(payload["enabled"]), payload.get("hz"))
        if method == "GET" and path == "/twin/status":
            return self.status()
        raise ValueError("Unknown endpoint")
