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

from brickcell.simulation.world import (  # noqa: E402
    BOWLS,
    COPIES,
    INSPECTION,
    OBJECTS,
    PARK,
    PARK_PLATES,
    World,
)
from brickcell.twin.sim_sync import (  # noqa: E402
    MAX_LIVE_HZ,
    TwinConflict,
    TwinSync,
    apply_state,
    is_parked,
    load_sim_config,
    plate_is_parked,
    resolve_sim_config,
    state_from_world,
)
from brickcell.twin.state import (  # noqa: E402
    ARM_JOINTS,
    BRICK_IDS,
    PARTS,
    ArmState,
    BrickState,
    PlateState,
    StationState,
    body_pose_from_mesh_pose,
    matrix_to_quat,
    part_of,
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
    pose = next(p for p in POSES[part_of(object_id).object_id]["poses"] if p["id"] == pose_id)
    rotation = rot_z(yaw) @ quat_to_matrix(pose["quat"])
    return BrickState(
        object_id,
        part_of(object_id).part,
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
    assert report["hidden"] == list(COPIES) and report["ok"]


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
    assert sorted(report["hidden"]) == sorted(["part-003", "part-004", *COPIES])
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
    assert set(config["park"]) == set(BRICK_IDS)
    assert config["park"]["part-001"] == [-1.05, 0.90]
    assert config["park_absent_plates"] == ["spare"]
    over = resolve_sim_config({"park": {"part-001-2": [-0.6, -0.9]}})
    assert over["park"]["part-001-2"] == [-0.6, -0.9] and over["park"]["part-002-2"]
    with pytest.raises(ValueError):
        resolve_sim_config({"park_absent_plates": ["good"]})  # no park spot for "good"
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


def test_apply_copies_observed_colour(world):
    from brickcell.twin.sim_sync import apply_state
    from brickcell.twin.state import BrickState, StationState

    red = [0.70, 0.05, 0.05]
    state = StationState(
        timestamp=1.0,
        bricks=[BrickState("part-002", "3011", [0.25, 0.0, 0.0], [1, 0, 0, 0], red)],
    )
    report = apply_state(world, state, {"relax_s": 0, "settle_s": 0})
    assert "part-002" in report["colors_changed"]
    rgba = world.model.geom("part-002-solid").rgba[:3]
    assert np.allclose(rgba, np.clip(np.array(red) * 1.08, 0, 1), atol=1e-6)
    # The same colour again: no change, so live mode does not rebuild the viewer each frame.
    again = apply_state(world, state, {"relax_s": 0, "settle_s": 0})
    assert again["colors_changed"] == {}
    off = apply_state(world, state, {"relax_s": 0, "settle_s": 0, "apply_colors": False})
    assert off["colors_changed"] == {}


# --------------------------------------------------------------------------- copies, spare plate


def _project(camera, points):
    """Pixels (u, v) of world points that are in front of the camera."""
    rotation, origin = np.array(camera["world_rotation_cv"]), np.array(camera["world_pos"])
    pc = (np.asarray(points) - origin) @ rotation
    pc = pc[pc[:, 2] > 0]
    return np.stack([camera["fx"] * pc[:, 0] / pc[:, 2] + camera["cx"],
                     camera["fy"] * pc[:, 1] / pc[:, 2] + camera["cy"]], axis=1)  # fmt: skip


def _extra_points():
    """Box corners around each parked copy and rim points of the parked spare plate."""
    points = []
    for oid in COPIES:
        x, y = PARK[oid]
        points += [
            [x + dx, y + dy, z] for dx in (-0.04, 0.04) for dy in (-0.04, 0.04) for z in (0, 0.05)
        ]
    x, y = PARK_PLATES["spare"]
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        points += [[x + 0.095 * np.cos(a), y + 0.095 * np.sin(a), z] for z in (0, 0.025)]
    return np.array(points)


def _demo_arm_poses(world):
    """TCP targets of the demo: start, above each brick, inspection, above each plate."""
    targets = [np.array([0.25, 0.0, 0.12])]
    targets += [np.array([0.25, (i - 1.5) * 0.065, z]) for i in range(4) for z in (0.16, 0.02)]
    targets += [
        INSPECTION,
        *[np.array([p[0], p[1], z]) for p in BOWLS.values() for z in (0.18, 0.04)],
    ]
    for target in targets:
        q = world.data.qpos[world.ik.qpos_adr].copy()
        for _ in range(100):
            q, _ = world.ik.solve(q, target, np.pi / 2)
        world.data.qpos[world.ik.qpos_adr] = q
        mujoco.mj_forward(world.model, world.data)
        yield target


def test_copies_exist_and_are_parked_out_of_view_at_reset():
    world = World(Path("artifacts/twin-test-world-copies"))
    try:
        m = world.model
        assert len(COPIES) == 12 and set(world.bricks) == set(BRICK_IDS)
        for oid, part in COPIES.items():
            solid, first = m.geom(f"{oid}-solid"), m.geom(f"{part}-solid")
            assert solid.dataid == first.dataid and np.allclose(solid.friction, first.friction)
            assert m.body(oid).mass == pytest.approx(m.body(part).mass)
            assert np.allclose(solid.rgba, first.rgba)
            assert m.geom(f"{oid}-mark").id >= 0
            assert world.is_parked(oid) and solid.contype == 0 and solid.conaffinity == 0
            assert np.allclose(world.position(oid)[:2], PARK[oid])
        for oid in OBJECTS:
            assert not world.is_parked(oid) and m.geom(f"{oid}-solid").contype == 1
        assert np.allclose(m.body("plate_spare").pos[:2], PARK_PLATES["spare"])
        spots = np.array(list(PARK.values()))
        gaps = np.linalg.norm(spots[:, None] - spots[None], axis=-1) + np.eye(len(spots))
        assert gaps.min() > 0.149  # parked bricks do not touch
        # The parked bodies hold still, with no contacts.
        world.hold(0.5)
        for oid in COPIES:
            assert np.linalg.norm(world.position(oid)[:2] - PARK[oid]) < 1e-6

        # Projection: no parked point is in an image (50 px margin), also the wrist camera at
        # the demo arm poses.
        def check_projection(tag):
            for c in world.cameras():
                if not c["enabled"]:
                    continue
                uv = _project(c, _extra_points())
                inside = (uv[:, 0] > -50) & (uv[:, 0] < c["width"] + 50)
                inside &= (uv[:, 1] > -50) & (uv[:, 1] < c["height"] + 50)
                assert not inside.any(), (tag, c["name"])

        # Render: each image is the same with the parked bodies far away (not in the scene).
        names = ["inspection", "overview", "overhead", "side", "wrist"]

        def renders():
            return {name: world.camera(name) for name in names}

        def without_extras(function):
            saved = world.data.qpos.copy()
            for oid in COPIES:
                a = m.joint(f"{oid}-joint").qposadr[0]
                world.data.qpos[a : a + 3] += [50, 0, 0]
            m.body("plate_spare").pos[0] += 50
            mujoco.mj_forward(m, world.data)
            try:
                return function()
            finally:
                m.body("plate_spare").pos[0] -= 50
                world.data.qpos[:] = saved
                mujoco.mj_forward(m, world.data)

        world.reset()
        check_projection("reset")
        with_extras = renders()
        reference = without_extras(renders)
        for name in names:
            assert np.array_equal(with_extras[name], reference[name]), name
        for target in _demo_arm_poses(world):
            check_projection(tuple(target))
            wrist = world.camera("wrist")
            assert np.array_equal(wrist, without_extras(lambda: world.camera("wrist"))), target
    finally:
        world.close()


def copies_state(with_spare=True):
    """Three copies of part-002, two copies of part-001 (one in the spare plate)."""
    bricks = [
        brick("part-002", "upright", 0.30, -0.08, yaw=0.2),
        brick("part-002-2", "upright", 0.30, 0.02, yaw=-0.3),
        brick("part-002-3", "upside_down", 0.13, -0.18, PLATE_FLOOR_TOP, support="plate_good"),
        brick("part-001", "upright", 0.40, 0.05),
        brick("part-001-3", "upright", 0.33, 0.23, PLATE_FLOOR_TOP, yaw=0.5, support="other"),
    ]
    for i, b in enumerate(bricks):
        b.color = [0.1 + 0.15 * i, 0.3, 0.6]
    plates = [PlateState("good", [0.13, -0.18, 0.0]), PlateState("defective", [0.08, 0.24, 0.0])]
    if with_spare:
        plates.append(PlateState("spare", [0.33, 0.24, 0.0]))
    return StationState(timestamp=time.time(), bricks=bricks, plates=plates)


def test_apply_copies_and_spare_plate(world):
    state = copies_state()
    report = apply_state(world, state, {"relax_s": 0.2, "settle_s": 1.0})
    assert report["ok"], report["flags"]
    for b in state.bricks:
        entry = report["bricks"][b.object_id]
        assert entry["drift_mm"] < 2.0 and entry["drift_deg"] < 2.0, (b.object_id, entry)
        assert not world.is_parked(b.object_id)
        # Colours per copy: each body has its own observed colour x gain.
        rgba = world.model.geom(f"{b.object_id}-solid").rgba[:3]
        assert np.allclose(rgba, np.clip(np.array(b.color) * 1.08, 0, 1), atol=1e-6)
    assert report["bricks"]["part-002-3"]["support"] == "plate_good"
    assert report["bricks"]["part-001-3"]["support"] == "plate_spare"
    assert np.allclose(body_xpos(world, "plate_spare")[:2], [0.33, 0.24])
    assert report["plates"]["spare"]["parked"] is False
    assert "spare" not in world.bowls  # not a sort target
    absent = [oid for oid in BRICK_IDS if oid not in {b.object_id for b in state.bricks}]
    assert sorted(report["hidden"]) == sorted(absent)
    for oid in absent:
        assert is_parked(world, oid) and np.allclose(body_xpos(world, oid)[:2], PARK[oid])

    # Round trip: the copies and the spare plate come back; the parked bricks do not.
    back = state_from_world(world)
    assert {b.object_id for b in back.bricks} == {b.object_id for b in state.bricks}
    assert {p.name for p in back.plates} == {"good", "defective", "spare"}
    again = StationState.from_json(back.to_json())
    apply_state(world, again, EXACT)
    for b in state_from_world(world).bricks:
        a = next(x for x in back.bricks if x.object_id == b.object_id)
        assert np.allclose(a.pos, b.pos, atol=1e-9)

    # Without the spare plate in the state: the spare plate is parked, the others stay.
    report = apply_state(world, copies_state(with_spare=False), EXACT)
    assert report["plates"]["spare"]["parked"] is True
    assert plate_is_parked(world, "spare", load_sim_config())
    assert "spare" not in {p.name for p in state_from_world(world).plates}
    # A sort plate that is not in the state keeps its place.
    state = copies_state(with_spare=False)
    state.plates = state.plates[:1]
    report = apply_state(world, state, EXACT)
    assert "defective" not in report["plates"]
    assert np.allclose(body_xpos(world, "plate_defective")[:2], [0.08, 0.24])
    # World.reset parks the copies and the spare plate again and keeps the demo as before.
    world.reset()
    assert all(world.is_parked(oid) for oid in COPIES)
    assert not any(world.is_parked(oid) for oid in OBJECTS)
    with pytest.raises(ValueError):
        world.action("pick", "part-002-2")  # the demo uses only the first copies


def test_set_brick_color_per_copy(world):
    before = world.model.geom("part-003-solid").rgba[:3].copy()
    rgb = world.set_brick_color("part-003-4", "#ff0000")
    assert rgb == (1.0, 0.0, 0.0)
    assert np.allclose(world.model.geom("part-003-4-solid").rgba[:3], rgb)
    assert np.allclose(world.model.geom("part-003-solid").rgba[:3], before)
    assert world.bricks["part-003-4"].rgba == rgb
