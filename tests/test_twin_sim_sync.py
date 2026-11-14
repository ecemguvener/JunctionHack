"""Station twin: apply a StationState to the MuJoCo World (MuJoCo environment only)."""

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("physicalai_mujoco_plugin")

from brickcell.simulation.world import World  # noqa: E402
from brickcell.twin.sim_sync import (  # noqa: E402
    MAX_LIVE_HZ,
    TwinConflict,
    TwinSync,
    apply_state,
    is_parked,
    load_sim_config,
    resolve_sim_config,
    state_from_world,
)
from brickcell.twin.state import (  # noqa: E402
    ARM_JOINTS,
    PARTS,
    ArmState,
    BrickState,
    PlateState,
    StationState,
    body_pose_from_mesh_pose,
    matrix_to_quat,
    quat_to_matrix,
    rot_z,
    rotation_angle_deg,
)

POSES = json.loads(Path("assets/duplo/stable_poses.json").read_text())["bricks"]
PLATE_FLOOR_TOP = 0.0031  # m; top of the KALAS floor disc
EXACT = {"relax_s": 0, "settle_s": 0}
ARM = {
    "shoulder_pan": 0.2,
    "shoulder_lift": -0.5,
    "elbow_flex": 0.8,
    "wrist_flex": 0.9,
    "wrist_roll": 0.1,
}


def brick(object_id, pose_id, x, y, floor=0.0, yaw=0.0, support="table"):
    pose = next(p for p in POSES[object_id]["poses"] if p["id"] == pose_id)
    rotation = rot_z(yaw) @ quat_to_matrix(pose["quat"])
    return BrickState(
        object_id,
        PARTS[object_id].part,
        [x, y, pose["rest_height"] + floor],
        matrix_to_quat(rotation).tolist(),
        [0.5, 0.5, 0.5],
        pose_id,
        support,
        0.9,
    )


def station_state(with_arm=True, all_bricks=True):
    """Four parts in stable poses (mesh frame); two of them on a moved plate."""
    bricks = [
        brick("part-001", "upright", 0.30, -0.06, yaw=0.3),
        brick("part-002", "upside_down", 0.32, 0.05),
        brick("part-003", "wide_side_py", 0.12, -0.15, PLATE_FLOOR_TOP, support="plate_good"),
        brick("part-004", "upright", 0.15, -0.21, PLATE_FLOOR_TOP, yaw=1.0, support="plate_good"),
    ]
    return StationState(
        timestamp=time.time(),
        bricks=bricks if all_bricks else bricks[:2],
        plates=[
            PlateState("good", [0.13, -0.18, 0.0], 0.9),
            PlateState("defective", [0.08, 0.24, 0.0]),
        ],
        arm=ArmState(dict(ARM), gripper=0.3, source="test") if with_arm else None,
    )


@pytest.fixture(scope="module")
def world():
    w = World(Path("artifacts/twin-test-world"))
    yield w
    w.close()


def body_xpos(world, name):
    return world.data.xpos[world.model.body(name).id].copy()


def test_apply_exact_poses_and_plates(world):
    state = station_state()
    report = apply_state(world, state, EXACT)
    for b in state.bricks:
        pos, quat = body_pose_from_mesh_pose(b.object_id, b.pos, b.quat)
        a = world.model.joint(f"{b.object_id}-joint").qposadr[0]
        assert np.allclose(world.data.qpos[a : a + 3], pos, atol=1e-6)
        assert np.allclose(
            quat_to_matrix(world.data.qpos[a + 3 : a + 7]), quat_to_matrix(quat), atol=1e-6
        )
        assert np.allclose(body_xpos(world, b.object_id), pos, atol=1e-6)
    assert np.allclose(body_xpos(world, "plate_good"), [0.13, -0.18, 0.0])
    assert np.allclose(body_xpos(world, "plate_defective"), [0.08, 0.24, 0.0])
    assert np.allclose(world.bowls["good"][:2], [0.13, -0.18])
    # Collision follows the moved plate: the floor geom is at the new place.
    floor = world.data.geom_xpos[world.model.geom("plate_good-floor").id]
    assert np.allclose(floor[:2], [0.13, -0.18])
    assert report["hidden"] == [] and report["ok"]


def test_settle_drift_and_bricks_stay_in_moved_plate(world):
    report = apply_state(world, station_state(), {"relax_s": 0.2, "settle_s": 1.0})
    for object_id, entry in report["bricks"].items():
        assert entry["drift_mm"] < 2.0, (object_id, entry)
        assert entry["drift_deg"] < 2.0, (object_id, entry)
        assert entry["stable_pose"] is not None
    centre = body_xpos(world, "plate_good")
    for object_id in ("part-003", "part-004"):
        assert report["bricks"][object_id]["support"] == "plate_good"
        assert np.linalg.norm(body_xpos(world, object_id)[:2] - centre[:2]) < 0.08
    assert report["ok"], report["flags"]


def test_penetration_and_gap_are_removed(world):
    state = station_state()
    state.bricks[0].pos[2] -= 0.002  # 2 mm into the table
    state.bricks[1].pos[2] += 0.002  # 2 mm above the table
    state.bricks[2].pos[2] -= 0.0015  # 1.5 mm into the plate floor
    report = apply_state(world, state, {})
    b = report["bricks"]
    assert b["part-001"]["penetration_mm"] == pytest.approx(2.0, abs=0.1)
    assert b["part-001"]["drift_z_mm"] == pytest.approx(2.0, abs=0.2)
    assert b["part-002"]["drift_z_mm"] == pytest.approx(-2.0, abs=0.2)
    assert b["part-003"]["drift_z_mm"] == pytest.approx(1.5, abs=0.2)
    mujoco.mj_forward(world.model, world.data)
    deepest = min((world.data.contact[i].dist for i in range(world.data.ncon)), default=0.0)
    assert deepest > -0.0003


def test_unstable_pose_is_flagged(world):
    state = station_state()
    tilt = quat_to_matrix([math.cos(math.radians(10)), math.sin(math.radians(10)), 0, 0])
    state.bricks[0].quat = matrix_to_quat(tilt).tolist()  # 20 deg tilt: not a resting pose
    state.bricks[0].pos[2] += 0.006
    report = apply_state(world, state, {})
    assert report["bricks"]["part-001"]["flags"]
    assert not report["ok"]


def test_arm_is_set_and_held(world):
    report = apply_state(world, station_state(), {})
    q = world.data.qpos[world.ik.qpos_adr]
    target = np.array([ARM[n] for n in ARM_JOINTS])
    assert np.degrees(np.abs(q - target)).max() < 1.0
    assert np.allclose(world.data.ctrl[world.actuators[:5]], target)
    assert world.grip == pytest.approx(0.3)
    world.hold(1.0)  # World.tick: the controller must hold the measured joints
    q = world.data.qpos[world.ik.qpos_adr]
    assert np.degrees(np.abs(q - target)).max() < 1.0
    assert report["arm"]["clamped"] == {}


def test_arm_clamps_are_reported(world):
    state = station_state()
    state.arm.joints["elbow_flex"] = 3.0
    report = apply_state(world, state, EXACT)
    clamp = report["arm"]["clamped"]["elbow_flex"]
    assert clamp["given"] == 3.0 and clamp["used"] < 3.0
    assert any("elbow_flex" in flag for flag in report["flags"])


def test_missing_bricks_are_parked_out_of_view(world):
    report = apply_state(world, station_state(all_bricks=False), {})
    assert sorted(report["hidden"]) == ["part-003", "part-004"]
    cameras = [c for c in world.cameras() if c["enabled"] and c["attach"] is None]
    for object_id in report["hidden"]:
        assert is_parked(world, object_id, load_sim_config())
        p = body_xpos(world, object_id)
        for c in cameras:
            pc = np.array(c["world_rotation_cv"]).T @ (p - np.array(c["world_pos"]))
            if pc[2] <= 0:
                continue
            u, v = c["fx"] * pc[0] / pc[2] + c["cx"], c["fy"] * pc[1] / pc[2] + c["cy"]
            assert not (0 <= u < c["width"] and 0 <= v < c["height"]), (object_id, c["name"])
    seen = state_from_world(world)
    assert sorted(b.object_id for b in seen.bricks) == ["part-001", "part-002"]


def test_state_from_world_round_trip(world):
    state = station_state()
    apply_state(world, state, EXACT)
    back = state_from_world(world)
    assert [b.object_id for b in back.bricks] == [b.object_id for b in state.bricks]
    for a, b in zip(state.bricks, back.bricks):
        assert np.allclose(a.pos, b.pos, atol=1e-6)
        assert rotation_angle_deg(quat_to_matrix(a.quat), quat_to_matrix(b.quat)) < 1e-3
        assert b.stable_pose == a.stable_pose and b.support == a.support
    plates = {p.name: p.pos[:2] for p in back.plates}
    assert plates["good"] == pytest.approx([0.13, -0.18])
    assert plates["defective"] == pytest.approx([0.08, 0.24])
    assert back.arm.joints == pytest.approx(ARM)
    # The JSON contract round trip, and the same sim state again.
    again = StationState.from_json(back.to_json())
    apply_state(world, again, EXACT)
    second = state_from_world(world)
    for a, b in zip(back.bricks, second.bricks):
        assert np.allclose(a.pos, b.pos, atol=1e-9)


def test_sim_config():
    config = load_sim_config()
    assert config["station_url"] == "http://127.0.0.1:8095"
    assert set(config["park"]) == set(PARTS)
    assert resolve_sim_config({"live_hz": 10})["live_hz"] == MAX_LIVE_HZ
    with pytest.raises(ValueError):
        resolve_sim_config({"settle": 1})
    with pytest.raises(ValueError):
        resolve_sim_config({"park": {"part-999": [0, 0]}})


def test_twin_requests_and_conflict(world):
    running = {"active": False}
    twin = TwinSync(world, EXACT, run_active=lambda: running["active"])
    state = station_state()
    report = twin.request("POST", "/twin/state", state.to_dict())
    assert set(report["bricks"]) == set(PARTS)
    status = twin.request("GET", "/twin/status", {})
    assert status["state"]["timestamp"] == state.timestamp and status["bricks"] == 4
    assert status["report"] is report and status["error"] is None and not status["live"]
    assert 0 <= status["state_age_s"] < 60
    with pytest.raises(ValueError):
        twin.request("POST", "/twin/state", {"timestamp": 1, "bricks": [{"object_id": "part-999"}]})
    running["active"] = True
    with pytest.raises(TwinConflict) as conflict:
        twin.request("POST", "/twin/state", state.to_dict())
    assert conflict.value.status == 409
    with pytest.raises(TwinConflict):
        twin.request("POST", "/twin/live", {"enabled": True})
    assert "TwinConflict" in twin.status()["error"]


def test_twin_sync_and_live(world):
    running = {"active": False}
    twin = TwinSync(world, {**EXACT, "live_hz": 2}, run_active=lambda: running["active"])
    base = station_state()
    clock = {"t": 100.0, "fetches": 0, "fresh": []}

    def fetch(fresh=False):
        clock["fetches"] += 1
        clock["fresh"].append(fresh)
        state = copy.deepcopy(base)
        state.timestamp = clock["t"]
        return state

    twin.fetch = fetch
    assert twin.request("POST", "/twin/sync", {"fresh": True})["timestamp"] == 100.0
    assert clock["fresh"] == [True]
    status = twin.request("POST", "/twin/live", {"enabled": True, "hz": 50})
    assert status["live"] and status["hz"] == MAX_LIVE_HZ
    deadline = time.monotonic() + 5
    while twin.status()["skipped_same_timestamp"] < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert twin.status()["skipped_same_timestamp"] >= 1  # same timestamp: not applied again
    clock["t"] = 101.0
    while twin.status()["applied"] < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert twin.status()["state"]["timestamp"] == 101.0
    running["active"] = True  # a run starts: live mode stops
    while twin.live and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not twin.live
    running["active"] = False
    twin.set_live(False)


def test_floor_is_visual_only(world):
    m = world.model
    floor, table = m.geom("floor").id, m.geom("table").id
    assert m.geom_contype[floor] == 0 and m.geom_conaffinity[floor] == 0
    assert m.geom_pos[floor][2] + m.geom_size[floor][2] == pytest.approx(0.0)  # top face at z = 0
    assert min(m.geom_size[floor][:2]) >= 1.0  # at least 2 m x 2 m
    assert m.geom_group[table] == 3 and m.geom_contype[table] == 1  # physics plane, not drawn
    assert np.allclose(m.geom_rgba[floor][:3], world.scene.floor_color)
    rgb = world.camera("overview").astype(float)
    corner = np.median(rgb[5:20, 5:20].reshape(-1, 3), axis=0)  # only the floor is in this corner
    r, g, b = corner
    assert r > g > b and r > 100  # light brown, as the real wooden table
