"""Verification tests for the SO-101 3D Block Localization & All-8-Corner Inspection stack."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from brickcell.block_inspector.armd_bridge import (
    SAFE_GRIPPER_MAX_RAW,
    SAFE_GRIPPER_MIN_RAW,
    ArmdFollowerBridge,
    ArmFaultError,
)
from brickcell.block_inspector.cogman import (
    CogManInspectionAgent,
    ExpectedAtomsMonitor,
    SymbolicState,
    compile_instruction,
    plan_strips_sequence,
)
from brickcell.block_inspector.geometry import (
    DUPLO_CATALOG,
    REPO_ROOT,
    T_BASE_WORLD,
    T_WORLD_BASE,
    corner_upper_hemisphere_mask,
    cv_cam_to_mujoco_cam,
    mujoco_cam_to_cv_cam,
    nominal_camera_rig,
    nominal_canted_wrist_cam_transform,
)
from brickcell.block_inspector.ledger import EightCornerCoverageLedger
from brickcell.block_inspector.runtime_bus import (
    CANONICAL_TOPICS,
    InProcessRuntimeBus,
    validate_babyros_topic,
)


def test_canonical_world_and_canted_wrist_transforms():
    # -Y_base -> +X_world and +X_base -> +Y_world (R_z(+90 deg))
    assert np.allclose(T_WORLD_BASE @ T_BASE_WORLD, np.eye(4), atol=1e-12)
    assert np.isclose(np.linalg.det(T_WORLD_BASE[:3, :3]), 1.0, atol=1e-12)
    assert np.allclose(T_WORLD_BASE[:3, :3] @ np.array([0.0, -1.0, 0.0]), [1.0, 0.0, 0.0])
    assert np.allclose(T_WORLD_BASE[:3, :3] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])

    # 25-deg canted wrist bracket must be proper right-handed SE(3) (det == +1)
    T_bc = nominal_canted_wrist_cam_transform(25.0)
    assert np.isclose(np.linalg.det(T_bc[:3, :3]), 1.0, atol=1e-12)
    assert np.allclose(T_bc[:3, :3].T @ T_bc[:3, :3], np.eye(3), atol=1e-12)
    # Optical axis +Z_cam tilts 25 deg toward +Y_Body3 (up) while pointing mainly along -Z_Body3
    z_cam_in_body3 = T_bc[:3, 2]
    assert np.isclose(z_cam_in_body3[2], -math.cos(math.radians(25.0)), atol=1e-12)

    # OpenCV <-> MuJoCo camera frame round-trip
    T_mj = cv_cam_to_mujoco_cam(T_bc)
    assert np.isclose(np.linalg.det(T_mj[:3, :3]), 1.0, atol=1e-12)
    assert np.allclose(mujoco_cam_to_cv_cam(T_mj), T_bc, atol=1e-12)


def test_duplo_catalog_assets_and_upper_hemisphere_partition():
    assert {
        "duplo_2x2_3437",
        "duplo_2x4_3011",
        "duplo_1x2x2_76371",
        "duplo_2x3_curved_top_2302",
    } <= set(DUPLO_CATALOG)
    for spec in DUPLO_CATALOG.values():
        assert spec.stl_path is not None and spec.stl_path.exists()
        assert spec.corners_local().shape == (8, 3)

    # Across all 6 resting cuboid face orientations, exactly 4 of 8 corners lie in upper hemisphere
    euler_cases = [
        (0.0, 0.0, 0.3),
        (math.pi, 0.0, -0.4),
        (0.5 * math.pi, 0.0, 0.2),
        (-0.5 * math.pi, 0.0, 1.1),
        (0.0, 0.5 * math.pi, -0.7),
        (0.0, -0.5 * math.pi, 0.5),
    ]
    for rx, ry, rz in euler_cases:
        T_wb = np.eye(4)
        T_wb[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()
        mask = corner_upper_hemisphere_mask(T_wb)
        assert int(np.sum(mask)) == 4


def test_eight_corner_ledger_predicates_dwell_and_symmetry_slip_guard():
    rig = nominal_camera_rig()
    spec = DUPLO_CATALOG["duplo_2x2_3437"]
    ledger = EightCornerCoverageLedger("block_0", spec, rig=rig)

    T_wb = np.eye(4)
    T_wb[:3, 3] = [0.20, 0.00, 0.5 * spec.dims_m[2]]
    ledger.set_initial_pose(T_wb)

    # Place wrist camera directly along corner 6 (+,+,+) outward bisector at 11.5 cm standoff
    c6 = ledger.corners_world()[6]
    b6 = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    p_cam = c6 + 0.115 * b6
    z_cam = -b6
    x_cam = np.cross(z_cam, np.array([0.0, 0.0, 1.0]))
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    T_world_cvcam = np.eye(4)
    T_world_cvcam[:3, :3] = np.column_stack([x_cam, y_cam, z_cam])
    T_world_cvcam[:3, 3] = p_cam
    T_world_body3 = T_world_cvcam @ np.linalg.inv(rig.wrist.T_parent_cvcam)

    # Crisp high-contrast checkerboard image so Laplacian sharpness >> 80
    checker = np.indices((480, 640)).sum(axis=0) // 6 % 2
    crisp_rgb = (checker * 220 + 20).astype(np.uint8)
    crisp_rgb = np.stack([crisp_rgb, crisp_rgb, crisp_rgb], axis=-1)
    blur_rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    zero_rays = np.zeros(8, dtype=np.float64)
    zero_vel = np.zeros(5, dtype=np.float64)

    # Blurry frame cannot increment dwell
    obs_blur = ledger.evaluate_camera_frame("wrist", blur_rgb, T_world_body3, zero_vel, zero_rays)
    ledger.update_step(wrist_obs=obs_blur)
    assert ledger.records[6].dwell_count == 0
    assert not ledger.records[6].inspected

    # High joint velocity (> 0.03 rad/s) blocks verification
    obs_fast = ledger.evaluate_camera_frame(
        "wrist", crisp_rgb, T_world_body3, np.array([0.25, 0, 0, 0, 0]), zero_rays
    )
    ledger.update_step(wrist_obs=obs_fast)
    assert ledger.records[6].dwell_count == 0

    # 3 consecutive valid crisp frames -> corner 6 becomes verified!
    for expected_dwell in (1, 2, 3):
        obs_ok = ledger.evaluate_camera_frame(
            "wrist",
            crisp_rgb,
            T_world_body3,
            zero_vel,
            zero_rays,
            anomaly_scorer=lambda roi: 0.42,
        )
        ledger.update_step(wrist_obs=obs_ok)
        assert ledger.records[6].dwell_count == expected_dwell
    assert ledger.records[6].inspected
    assert np.isclose(ledger.records[6].raw_anomaly_distance, 0.42)

    # Known 90-deg yaw manipulation locks cleanly without resetting c_6;
    # unexpected 90-deg D4 slip triggers conservative reset!
    T_grasp = np.eye(4)
    T_place = np.eye(4)
    T_place[:3, :3] = Rotation.from_euler("z", 0.5 * math.pi).as_matrix()
    T_prior = ledger.propagate_manipulation_prior(T_grasp, T_place)
    assert ledger.commit_post_manipulation_pose(T_prior, icp_fitness=0.92, patch_ncc=0.88)
    assert ledger.records[6].inspected

    # Simulate unmodeled 90-deg slip in jaws (> 35 deg from prior) -> conservative reset
    T_slipped = T_prior.copy()
    T_slipped[:3, :3] = Rotation.from_euler("z", 0.5 * math.pi).as_matrix() @ T_prior[:3, :3]
    ledger.propagate_manipulation_prior(np.eye(4), np.eye(4))
    assert not ledger.commit_post_manipulation_pose(T_slipped, icp_fitness=0.92, patch_ncc=0.88)
    assert not ledger.records[6].inspected


def test_cogman_strips_planner_deadlock_free_progression():
    # Default plate_spin_sort workflow: Localize -> Pick -> Plate CCW Spin -> Sort (good vs defective) -> Park
    goal_sort = compile_instruction("investigate this block", block_id="block_0")
    assert goal_sort.mode == "plate_spin_sort"

    ps0 = SymbolicState(
        block_id="block_0",
        hand_empty=True,
        block_localized=False,
        block_on_table=True,
        block_held=False,
        block_at_rest=False,
    )
    assert plan_strips_sequence(ps0, goal_sort)[0].name == "LocalizeBlock"

    ps1 = SymbolicState(
        block_id="block_0",
        hand_empty=True,
        block_localized=True,
        block_on_table=True,
        block_held=False,
        block_at_rest=False,
    )
    assert plan_strips_sequence(ps1, goal_sort)[0].name == "PickBlock"

    ps2 = SymbolicState(
        block_id="block_0",
        hand_empty=False,
        block_localized=True,
        block_on_table=False,
        block_held=True,
        block_at_rest=False,
        plate_scan_completed=False,
    )
    assert plan_strips_sequence(ps2, goal_sort)[0].name == "PresentAndSpinAtPlateCamera"

    ps3_clean = SymbolicState(
        block_id="block_0",
        hand_empty=False,
        block_localized=True,
        block_on_table=False,
        block_held=True,
        block_at_rest=False,
        plate_scan_completed=True,
        anomaly_flagged=False,
    )
    op_clean = plan_strips_sequence(ps3_clean, goal_sort)[0]
    assert op_clean.name == "SortBlockToPile" and op_clean.params["pile"] == "good"

    ps3_anom = SymbolicState(
        block_id="block_0",
        hand_empty=False,
        block_localized=True,
        block_on_table=False,
        block_held=True,
        block_at_rest=False,
        plate_scan_completed=True,
        anomaly_flagged=True,
    )
    op_anom = plan_strips_sequence(ps3_anom, goal_sort)[0]
    assert op_anom.name == "SortBlockToPile" and op_anom.params["pile"] == "defective"

    ps4_sorted = SymbolicState(
        block_id="block_0",
        hand_empty=True,
        block_localized=True,
        block_on_table=False,
        block_held=False,
        block_at_rest=False,
        plate_scan_completed=True,
        anomaly_flagged=False,
        sorted_pile="good",
    )
    assert plan_strips_sequence(ps4_sorted, goal_sort)[0].name == "ParkAtRest"

    # Legacy eight_corner mode remains deadlock-free as well
    goal = compile_instruction("investigate this block", block_id="block_0", mode="eight_corner")

    # Step 1: Unlocalized -> LocalizeBlock
    s0 = SymbolicState(
        block_id="block_0",
        hand_empty=True,
        block_localized=False,
        block_on_table=True,
        block_held=False,
        block_at_rest=False,
        corner_upper=(False,) * 8,
        corner_wrist_reachable=(False,) * 8,
        corner_inspected=(False,) * 8,
    )
    assert plan_strips_sequence(s0, goal)[0].name == "LocalizeBlock"

    # Step 2: 2 reachable upper corners + 2 far unreachable upper corners -> Inspect reachable first
    s1 = SymbolicState(
        block_id="block_0",
        hand_empty=True,
        block_localized=True,
        block_on_table=True,
        block_held=False,
        block_at_rest=False,
        corner_upper=(False, True, False, True, False, True, False, True),
        corner_wrist_reachable=(False, True, False, True, False, False, False, False),
        corner_inspected=(False,) * 8,
    )
    p1 = plan_strips_sequence(s1, goal)
    assert [op.name for op in p1] == ["InspectExposedCorner", "InspectExposedCorner"]

    # Step 3: Reachable upper corners done, far upper corners remain -> ReorientBlockYaw(pi)
    s2 = SymbolicState(
        block_id="block_0",
        hand_empty=True,
        block_localized=True,
        block_on_table=True,
        block_held=False,
        block_at_rest=False,
        corner_upper=(False, True, False, True, False, True, False, True),
        corner_wrist_reachable=(False, True, False, True, False, False, False, False),
        corner_inspected=(False, True, False, True, False, False, False, False),
    )
    assert plan_strips_sequence(s2, goal)[0].name == "ReorientBlockYaw"

    # Step 4: All 4 upper corners done, bottom 4 remain -> GraspPresentAndFlipBlock
    s3 = SymbolicState(
        block_id="block_0",
        hand_empty=True,
        block_localized=True,
        block_on_table=True,
        block_held=False,
        block_at_rest=False,
        corner_upper=(False, True, False, True, False, True, False, True),
        corner_wrist_reachable=(False, True, False, True, False, True, False, True),
        corner_inspected=(False, True, False, True, False, True, False, True),
    )
    assert plan_strips_sequence(s3, goal)[0].name == "GraspPresentAndFlipBlock"

    # Step 5: All 8 corners inspected -> ParkAtRest
    s4 = SymbolicState(
        block_id="block_0",
        hand_empty=True,
        block_localized=True,
        block_on_table=True,
        block_held=False,
        block_at_rest=False,
        corner_upper=(True, False, True, False, True, False, True, False),
        corner_wrist_reachable=(True, False, True, False, True, False, True, False),
        corner_inspected=(True,) * 8,
    )
    assert plan_strips_sequence(s4, goal)[0].name == "ParkAtRest"

    mon = ExpectedAtomsMonitor(seed=3)
    pert = mon.resample_cone_perturbation(2)
    assert pert.shape == (3,) and np.all(np.abs(pert) <= 0.15)


def test_runtime_bus_and_deployment_manifests():
    with pytest.raises(ValueError, match="should not start with '/'"):
        validate_babyros_topic("/so101/arm/state")

    for t in CANONICAL_TOPICS.values():
        assert validate_babyros_topic(t) == t

    bus = InProcessRuntimeBus()
    received = []
    bus.publish(CANONICAL_TOPICS["block_state"], {"localized": True}, transient_local=True)
    bus.subscribe(
        CANONICAL_TOPICS["block_state"], lambda m: received.append(m), transient_local=True
    )
    assert received == [{"localized": True}]

    wendy = json.loads((REPO_ROOT / "wendy.json").read_text(encoding="utf-8"))
    assert wendy["isolation"] == "shared-ipc"
    allowlist = wendy["services"]["camera-pub"]["entitlements"][0]["allowlist"]
    assert "/dev/video44" in allowlist
    assert not any(f"/dev/video{k}" in allowlist for k in range(32))

    qualia = json.loads(
        (REPO_ROOT / "configs" / "qualia" / "inspection_task.json").read_text(encoding="utf-8")
    )
    assert qualia["displayName"] == "Investigate Block"
    assert "all 8" in qualia["draft"]["description"]


def test_armd_bridge_safety_clamps_and_fake_server_roundtrip():
    calls = []
    bridge = ArmdFollowerBridge(
        request_fn=lambda method, path, body: calls.append((method, path, body)) or {"ok": True}
    )
    # Gripper raw tick floor is strictly >= 1339 even for negative/zero input
    assert bridge.gripper_pct_to_safe_ticks(-2.0) == SAFE_GRIPPER_MIN_RAW
    assert bridge.gripper_pct_to_safe_ticks(0.0) == SAFE_GRIPPER_MIN_RAW
    assert bridge.gripper_pct_to_safe_ticks(1.0) == SAFE_GRIPPER_MAX_RAW

    # Speed 0 is upgraded to >= 1 so armd never rejects with 400 or drives max servo speed
    payload = bridge.build_positions_payload(np.zeros(5), gripper_pct=0.0, speed=0)
    assert payload["joints"]["shoulder_pan"]["speed"] == 1
    assert payload["joints"]["gripper"]["position"] == SAFE_GRIPPER_MIN_RAW

    # 4096 vs 4095 LeRobot 0.7 scale round-trip
    q_orig = np.array([0.2, -0.4, 0.5, -0.3, 0.7])
    act6 = bridge.q_to_lerobot_action6(q_orig, 0.65)
    q_back, g_back = bridge.lerobot_action6_to_q(act6)
    assert np.allclose(q_orig, q_back, atol=1e-10)
    assert np.isclose(g_back, 0.65, atol=1e-10)

    # Partial single-servo dropout returns None cleanly without raising
    dropout_msg = {
        "joints": {
            "shoulder_pan": {"position": None, "online": False, "healthy": True},
            "shoulder_lift": {"position": 2048, "online": True, "healthy": True},
            "elbow_flex": {"position": 2048, "online": True, "healthy": True},
            "wrist_flex": {"position": 2048, "online": True, "healthy": True},
            "wrist_roll": {"position": 2048, "online": True, "healthy": True},
            "gripper": {"position": 2000, "online": True, "healthy": True},
        }
    }
    assert bridge.decode_state_message(dropout_msg) is None

    # Blocking joint fault immediately triggers POST /arms/follower/stop {'mode': 'hold'}
    fault_msg = {
        "joints": {
            "shoulder_pan": {"position": 2048, "online": True, "healthy": True},
            "shoulder_lift": {"position": 2048, "online": True, "healthy": True},
            "elbow_flex": {"position": 2048, "online": True, "healthy": True},
            "wrist_flex": {"position": 2048, "online": True, "healthy": True},
            "wrist_roll": {"position": 2048, "online": True, "healthy": True},
            "gripper": {
                "position": 1339,
                "online": True,
                "healthy": False,
                "errors": ["overload"],
            },
        }
    }
    calls.clear()
    with pytest.raises(ArmFaultError, match="overload"):
        bridge.decode_state_message(fault_msg)
    assert calls == [("POST", "/arms/follower/stop", {"mode": "hold"})]

    # Full round-trip against armd Pydantic schema + Arm + FakeBus (socket-free for sandbox)
    pytest.importorskip("fastapi")
    from armd.arm import Arm
    from armd.config import config_from_dict
    from armd.fake import FakeBus
    from armd.server import PositionsBody

    cfg = config_from_dict(
        {
            "server": {"host": "127.0.0.1", "port": 8765, "poll_hz": 100, "admin": False},
            "arms": {
                "follower": {
                    "port": "/dev/fake-follower",
                    "allow_motion": True,
                    "allow_spin": False,
                    "max_speed": 2000,
                    "default_speed": 400,
                    "default_acc": 30,
                    "on_shutdown": "hold",
                    "block_on_errors": ["overload", "temperature", "current", "sensor"],
                    "joints": {
                        "shoulder_pan": {"id": 1},
                        "shoulder_lift": {"id": 2},
                        "elbow_flex": {"id": 3},
                        "wrist_flex": {"id": 4},
                        "wrist_roll": {"id": 5},
                        "gripper": {"id": 6, "min": 1339, "max": 2829},
                    },
                }
            },
        }
    )
    acfg = cfg.arms["follower"]
    arm = Arm(
        "follower",
        acfg,
        FakeBus(acfg.port, acfg.baudrate, ids=[1, 2, 3, 4, 5, 6]),
        poll_hz=100,
    )
    arm.start()
    try:

        def _direct_armd(method: str, path: str, body: dict | None):
            if method == "POST" and path == "/arms/follower/positions":
                parsed = PositionsBody.model_validate(body)
                targets = {k: v.model_dump(exclude_none=True) for k, v in parsed.joints.items()}
                return arm.move_many(
                    targets,
                    speed=parsed.speed,
                    acc=parsed.acc,
                    wait=parsed.wait,
                    timeout=parsed.timeout,
                    tolerance=parsed.tolerance,
                )
            if method == "GET" and path.startswith("/arms/follower/state"):
                return arm.state(fresh="fresh=true" in path)
            if method == "POST" and path == "/arms/follower/stop":
                return arm.stop(mode=(body or {}).get("mode", "hold"))
            raise AssertionError(f"Unexpected call: {method} {path}")

        live_bridge = ArmdFollowerBridge(request_fn=_direct_armd)
        res = live_bridge.send_command(np.zeros(5), gripper_pct=0.5, wait=True)
        assert "targets" in res and res["reached"] is True
        st = live_bridge.read_follower_state(fresh=True)
        assert st is not None and st.healthy
        assert st.gripper_ticks >= SAFE_GRIPPER_MIN_RAW
    finally:
        arm.shutdown()


def test_frax_kinematics_lambert_antipodal_guard_and_lookat_ik(tmp_path: Path):
    pytest.importorskip("frax")
    import jax.numpy as jnp
    from so101_kinematics import CalibratedSO101Kinematics

    from brickcell.block_inspector.frax_ik import (
        SO101FraxIK,
        export_calibrated_frax_assets,
        so101_antipodal_safe_lambert_2d,
    )

    urdf_path, spheres_path = export_calibrated_frax_assets(tmp_path)
    assert urdf_path.exists() and spheres_path.exists()

    ik = SO101FraxIK(assets_dir=tmp_path)
    ref_kin = CalibratedSO101Kinematics("follower")

    # Verify Frax Manipulator FK & Jacobian agree with CalibratedSO101Kinematics to < 1e-8
    q_test = np.array([0.25, -0.55, 0.70, -0.35, 0.80], dtype=np.float64)
    T_ref_base = ref_kin.fk(q_test)
    J_ref_base = ref_kin.jacobian(q_test)

    T_frax_body3 = np.asarray(ik.robot.ee_transform(jnp.asarray(q_test)), dtype=np.float64)
    J_frax_base = np.asarray(ik.robot.ee_jacobian(jnp.asarray(q_test)), dtype=np.float64)
    assert np.allclose(T_frax_body3, T_ref_base, atol=1e-8)
    assert np.allclose(J_frax_base, J_ref_base, atol=1e-8)

    # Verify South-pole guard on Lambert projection at exact antipodal pointing (-Z_cam)
    anti = np.asarray(so101_antipodal_safe_lambert_2d(jnp.array([0.0, 0.0, -1.0])))
    assert np.all(np.isfinite(anti))
    assert float(np.linalg.norm(anti)) >= 1.999

    # Solve 5-DOF Look-At IK across resting DUPLO 2x2 block upper corners
    spec = DUPLO_CATALOG["duplo_2x2_3437"]
    T_wb = np.eye(4)
    T_wb[:3, 3] = [0.205, 0.00, 0.5 * spec.dims_m[2]]
    sols = ik.evaluate_upper_corners_reachability(T_wb, spec)
    assert len(sols) == 4
    reachable_near = [s for s in sols.values() if s.reachable]
    assert len(reachable_near) >= 2
    for s in reachable_near:
        assert 0.08 <= s.standoff_m <= 0.165
        assert s.pointing_cos > math.cos(math.radians(15.0))


def test_end_to_end_mujoco_4_plate_camera_ccw_spin_and_sort(tmp_path: Path):
    mj = pytest.importorskip("mujoco")
    pytest.importorskip("frax")
    assert mj.__version__ == "4.0.1"

    from brickcell.block_inspector.twin import (
        DEFECTIVE_PILE_XY,
        GOOD_PILE_XY,
        SO101InspectionTwin,
    )

    twin = SO101InspectionTwin(
        block_spec="duplo_2x2_3437",
        scene_dir=tmp_path / "mjcf",
        initial_block_xy_yaw=(0.205, 0.00, 0.22),
        show_viewer=False,
        realtime_pacing=False,
    )
    try:
        # 1. Verify published_priors_verified.json dynamics + configs/sim-scene.yaml cameras
        assert np.isclose(T_WORLD_BASE[2, 3], 0.02932886598977181, atol=1e-7)
        assert np.allclose(twin.model.dof_damping[:5], 0.6, atol=1e-6)
        assert np.allclose(twin.model.dof_armature[:5], 0.028, atol=1e-6)
        assert np.allclose(twin.model.dof_frictionloss[:5], 0.052, atol=1e-6)
        assert (twin.rig.realsense_1.width, twin.rig.realsense_1.height) == (640, 480)
        assert (twin.rig.realsense_2.width, twin.rig.realsense_2.height) == (640, 480)
        assert (twin.rig.wrist.width, twin.rig.wrist.height) == (640, 360)
        assert np.allclose(
            twin.rig.realsense_2.T_parent_cvcam[:3, 3], [0.3699, -0.4686, 0.2189], atol=1e-4
        )

        # 2. Strict anti-cheat guard: forbid any direct block qpos write during skills!
        real_set_pose = twin._set_settled_block_pose

        def _forbid_teleport(*_args, **_kwargs):
            raise AssertionError("Direct qpos block teleport is strictly forbidden in skills!")

        # 3. Safety guard: missing threshold with anomaly_scorer must refuse to sort
        twin.anomaly_scorer = lambda _roi: 0.12
        twin.anomaly_threshold = None
        twin._set_settled_block_pose = _forbid_teleport  # type: ignore[method-assign]
        agent = CogManInspectionAgent(executor=twin, ledger=twin.ledger)
        untrusted_rep = agent.run(instruction="investigate this block", block_id="block_0")
        assert not untrusted_rep.succeeded
        assert twin.sorted_pile is None

        # 4. Clean block (no anomaly message) -> Pick -> Plate-Cam CCW Spin -> Good Pile!
        twin.anomaly_scorer = None
        twin.reset_episode_state()
        twin.reset_arm_to_home()
        real_set_pose(
            np.array([0.205, 0.00, 0.5 * twin.block_spec.dims_m[2]]),
            Rotation.from_euler("z", 0.22).as_matrix(),
        )
        twin._set_settled_block_pose = _forbid_teleport  # type: ignore[method-assign]
        clean_rep = agent.run(
            instruction="investigate this block",
            block_id="block_0",
            output_dir=tmp_path / "reports",
        )
        assert clean_rep.succeeded, f"Trace: {json.dumps(clean_rep.executed_trace, indent=2)}"
        assert clean_rep.sorted_pile == "good"
        assert not clean_rep.anomaly_flagged
        assert twin.plate_scan_valid_frames >= 8
        assert len(twin.plate_scan_keyframes) == 4
        for kf in twin.plate_scan_keyframes:
            assert cv2.cvtColor(kf, cv2.COLOR_RGB2GRAY).std() > 4.0
        p_good = twin.ground_truth_block_pose()[:3, 3]
        assert float(np.linalg.norm(p_good[:2] - GOOD_PILE_XY)) <= 0.080
        assert clean_rep.report_json_path is not None and Path(clean_rep.report_json_path).exists()
        assert clean_rep.report_html_path is not None and Path(clean_rep.report_html_path).exists()

        # 5. Incoming anomaly message mid-CCW-spin -> Defective Pile!
        twin.reset_episode_state()
        twin.reset_arm_to_home()
        real_set_pose(
            np.array([0.205, 0.00, 0.5 * twin.block_spec.dims_m[2]]),
            Rotation.from_euler("z", -0.18).as_matrix(),
        )
        twin.anomaly_message_fn = lambda _tw, _k, q4, _rgb: (
            "mid_spin_defect_msg" if q4 >= 0.0 else None
        )
        twin._set_settled_block_pose = _forbid_teleport  # type: ignore[method-assign]
        anom_rep = agent.run(instruction="investigate this block", block_id="block_0")
        assert anom_rep.succeeded, f"Trace: {json.dumps(anom_rep.executed_trace, indent=2)}"
        assert anom_rep.anomaly_flagged is True
        assert anom_rep.sorted_pile == "defective"
        p_def = twin.ground_truth_block_pose()[:3, 3]
        assert float(np.linalg.norm(p_def[:2] - DEFECTIVE_PILE_XY)) <= 0.080
    finally:
        twin.close()


def test_three_plates_eight_clustered_mixed_duplo_blocks_openvino_sort(tmp_path: Path):
    pytest.importorskip("mujoco")
    pytest.importorskip("frax")
    pytest.importorskip("openvino")

    from brickcell.block_inspector.geometry import station_plates_config
    from brickcell.block_inspector.openvino_anomaly import ensure_default_plate_openvino_detector
    from brickcell.block_inspector.twin import (
        DEFECTIVE_PILE_XY,
        GOOD_PILE_XY,
        SOURCE_PILE_XY,
        SO101InspectionTwin,
    )

    plates = station_plates_config()
    assert set(plates) == {"source", "good", "defective"}
    assert np.allclose(plates["source"].rgba[:3], [0.28, 0.82, 0.80], atol=1e-3)
    assert np.allclose(plates["source"].pos_m, [0.23, 0.00, 0.0], atol=1e-4)

    detector = ensure_default_plate_openvino_detector(tmp_path / "ov_detector")
    assert detector.threshold is not None

    twin = SO101InspectionTwin(
        num_blocks=8,
        cluster_seed=42,
        scene_dir=tmp_path / "mjcf_cluster8",
        show_viewer=False,
        realtime_pacing=False,
        anomaly_scorer=lambda roi: detector.predict(roi).score,
        anomaly_threshold=float(detector.threshold.value),
    )
    try:
        # Verify all 3 KALAS plates and all 4 DUPLO brick types are present in the scene
        for pname in ("plate_source", "plate_good", "plate_defective"):
            assert twin.model.body(pname).id >= 0
        spawned_types = {b.spec.name for b in twin.spawned_blocks}
        assert spawned_types == set(DUPLO_CATALOG.keys())
        expected_defects = sum(1 for b in twin.spawned_blocks if b.has_visual_defect)
        expected_good = 8 - expected_defects
        assert 1 <= expected_defects <= 7

        # Verify all 8 blocks start clustered inside the aquamarine source plate
        for b in twin.spawned_blocks:
            p0 = twin.data.xpos[twin.block_body_ids[b.block_id]]
            assert float(np.linalg.norm(p0[:2] - SOURCE_PILE_XY)) <= 0.082

        # Strict anti-cheat guard: forbid any direct block qpos teleport during execution!
        def _forbid_teleport(*_args, **_kwargs):
            raise AssertionError("Direct qpos block teleport is strictly forbidden!")

        twin._set_settled_block_pose = _forbid_teleport  # type: ignore[method-assign]

        agent = CogManInspectionAgent(executor=twin, ledger=twin.ledger)
        rep = agent.run(
            instruction="pick and classify all 8 blocks from source plate",
            mode="clear_source_plate",
            output_dir=tmp_path / "cluster_reports",
        )
        assert rep.succeeded, f"Trace: {json.dumps(rep.executed_trace[-10:], indent=2)}"
        assert twin.source_plate_cleared is True
        assert sum(rep.sorted_counts.values()) == 8
        assert rep.sorted_counts["good"] == expected_good
        assert rep.sorted_counts["defective"] == expected_defects
        assert len(rep.batch_sorted_records) == 8

        # Verify physical end positions of all 8 rigid bodies match their ground-truth defect status
        for b in twin.spawned_blocks:
            p_end = twin.data.xpos[twin.block_body_ids[b.block_id]]
            target_xy = DEFECTIVE_PILE_XY if b.has_visual_defect else GOOD_PILE_XY
            assert float(np.linalg.norm(p_end[:2] - target_xy)) <= 0.088
    finally:
        twin.close()


def test_core_supervisor_boundaries_and_openvino_manifest_partitions(tmp_path: Path):
    pytest.importorskip("mujoco")
    pytest.importorskip("frax")
    pytest.importorskip("openvino")

    from brickcell.block_inspector.cell_adapter import TwinPlateFrameSource, TwinRobotAdapter
    from brickcell.block_inspector.openvino_anomaly import (
        generate_synthetic_plate_anomaly_dataset,
        train_calibrate_and_evaluate_plate_detector,
    )
    from brickcell.block_inspector.twin import SO101InspectionTwin, SpawnedBlockInfo
    from brickcell.controller.robot import ActionStatus
    from brickcell.controller.supervisor import ControllerConfig, Supervisor
    from brickcell.inspection.images import QualityConfig
    from brickcell.inspection.service import InspectionService
    from brickcell.inspection.types import Decision

    manifest_path = generate_synthetic_plate_anomaly_dataset(tmp_path / "dataset")
    detector, eval_summary = train_calibrate_and_evaluate_plate_detector(
        manifest_path, tmp_path / "model"
    )
    assert eval_summary["score_type"] == "raw_patchcore_distance_not_probability"
    assert eval_summary["partition_counts"]["train/good"] == 32
    assert eval_summary["test_confusion"]["tp"] >= 7
    assert eval_summary["test_confusion"]["fn"] == 0

    spec = DUPLO_CATALOG["duplo_2x2_3437"]
    twin = SO101InspectionTwin(
        block_spec=spec,
        spawned_blocks=[
            SpawnedBlockInfo("block_0", spec, (0.23, 0.00, 0.15), has_visual_defect=True)
        ],
        scene_dir=tmp_path / "mjcf_sup",
        show_viewer=False,
        realtime_pacing=False,
    )
    try:
        robot_adapter = TwinRobotAdapter(twin)
        assert robot_adapter.mode == "mujoco_simulation_not_physical"
        frame_source = TwinPlateFrameSource(robot_adapter, tmp_path / "captured_frames")
        inspector = InspectionService(
            detector=detector,
            quality=QualityConfig(max_outside_fraction=0.12),
            map_dir=tmp_path / "maps",
        )
        sup = Supervisor(
            robot=robot_adapter,
            source=frame_source,
            inspector=inspector,
            config=ControllerConfig(action_timeout_s=25.0, poll_interval_s=0.001),
        )
        summary = sup.run(["sim-obj-001"])
        assert summary["completed_cycles"] == 1
        assert summary["physical_success_verified"] is False
        assert summary["errors"] == 0
        cyc = summary["cycles"][0]
        assert cyc.complete is True
        assert cyc.decision == Decision.DEFECTIVE
        # Confirm non-blocking token lifecycle completed cleanly
        assert all(st == ActionStatus.SUCCEEDED for st in robot_adapter._statuses.values())
    finally:
        twin.close()


def test_multinode_babyros_runtime_graph_and_xpolicylab_checkpoint_roundtrip(tmp_path: Path):
    pytest.importorskip("mujoco")
    pytest.importorskip("frax")
    pytest.importorskip("openvino")

    from brickcell.block_inspector.openvino_anomaly import ensure_default_plate_openvino_detector
    from brickcell.block_inspector.policy_skills import (
        ChunkedMLPPolicy,
        LocalXPolicyWebSocketServer,
        RemoteXPolicyClient,
        _encode_policy_features,
        export_lerobot_v3_metadata,
        pack_xpolicylab_observation,
    )
    from brickcell.block_inspector.runtime_nodes import (
        CameraPubNode,
        CogManInspectorNode,
        FraxMujocoTwinNode,
        PerceptionAndAnomalyNode,
    )
    from brickcell.block_inspector.twin import (
        DEFECTIVE_PILE_XY,
        SO101InspectionTwin,
        SpawnedBlockInfo,
    )

    # 1. Multi-node BabyROS closed-loop sort (`CameraPubNode -> PerceptionAndAnomalyNode -> TwinNode <-> CogManNode`)
    bus = InProcessRuntimeBus()
    detector = ensure_default_plate_openvino_detector(tmp_path / "ov_bus_det")
    spec = DUPLO_CATALOG["duplo_2x2_3437"]
    twin = SO101InspectionTwin(
        block_spec=spec,
        spawned_blocks=[
            SpawnedBlockInfo("block_0", spec, (0.23, 0.00, 0.20), has_visual_defect=True)
        ],
        scene_dir=tmp_path / "mjcf_bus",
        show_viewer=False,
        realtime_pacing=False,
    )
    cam_node = CameraPubNode(bus=bus, source=twin)
    perc_node = PerceptionAndAnomalyNode(bus=bus, detector=detector)
    twin_node = FraxMujocoTwinNode(bus=bus, twin=twin, camera_pub=cam_node)
    cogman_node = CogManInspectorNode(bus=bus, hud_port=None)
    try:
        cam_node.publish_step()
        cluster_info = perc_node.localize_cluster_once()
        assert cluster_info["num_candidates"] == 1
        rep = cogman_node.run_episode(
            instruction="pick and classify all 8 blocks from source plate",
            mode="clear_source_plate",
        )
        assert rep.succeeded is True
        assert perc_node.anomaly_alerts_published >= 1
        assert rep.sorted_counts == {"good": 0, "defective": 1}
        p_end = twin.ground_truth_block_pose()[:3, 3]
        assert float(np.linalg.norm(p_end[:2] - DEFECTIVE_PILE_XY)) <= 0.088
    finally:
        cogman_node.close()
        twin_node.close()
        perc_node.close()
        twin.close()
        bus.close()

    # 2. ChunkedMLPPolicy checkpoint + lerobot_v3 metadata + XPolicyLab WebSocket round-trip
    rng = np.random.default_rng(7)
    X_syn = np.vstack(
        [
            _encode_policy_features(
                rng.normal(0.0, 0.2, size=5),
                0.5,
                k / 15.0,
                float(k % 2),
                np.array([0.23, 0.0, 0.016]),
                0.1,
            )
            for k in range(16)
        ]
    )
    Y_syn = rng.normal(0.0, 0.1, size=(16, 8 * 6))
    pol = ChunkedMLPPolicy(chunk_horizon=8, seed=7)
    pol.fit(X_syn, Y_syn)
    ckpt_file = pol.save_checkpoint(tmp_path / "skill_lab" / "chunk_bc_test.npz")
    pol_loaded = ChunkedMLPPolicy.load_checkpoint(ckpt_file)
    assert np.allclose(pol.predict_chunk(X_syn[0]), pol_loaded.predict_chunk(X_syn[0]), atol=1e-9)

    ep_dir = tmp_path / "skill_lab" / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "episode_0000.json").write_text('{"episode_index": 0}\n', encoding="utf-8")
    info_p, man_p = export_lerobot_v3_metadata(
        tmp_path / "skill_lab",
        "duplo_2x2_3437",
        [{"episode_index": 0, "tasks": ["test"], "length": 40}],
    )
    assert json.loads(info_p.read_text(encoding="utf-8"))["codebase_version"] == "v3.0"
    assert (
        len(json.loads(man_p.read_text(encoding="utf-8"))["kept_episodes"][0]["episode_sha256"])
        == 64
    )

    obs = pack_xpolicylab_observation(
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.zeros(5, dtype=np.float32),
        0.5,
    )
    ws_srv = LocalXPolicyWebSocketServer(policy=pol_loaded)
    wire_cli = RemoteXPolicyClient(transport_fn=ws_srv.handle_packed_request)
    wire_chunk = wire_cli.infer_chunk(obs, np.zeros(5), 0.5, horizon=8)
    assert wire_chunk.shape == (8, 6)
    assert wire_cli.last_used_fallback is False
    assert ws_srv.requests_served == 1

    # Also verify stationary-hold safety fallback when remote endpoint is unreachable/sandboxed
    fb_cli = RemoteXPolicyClient(ws_url="ws://127.0.0.1:19999", timeout_s=0.05)
    fb_chunk = fb_cli.infer_chunk(obs, np.array([0.1, -0.2, 0.3, -0.4, 0.5]), 0.65, horizon=8)
    assert fb_chunk.shape == (8, 6)
    assert fb_cli.last_used_fallback is True
    assert np.allclose(fb_chunk[0], [0.1, -0.2, 0.3, -0.4, 0.5, 0.65])


def test_real2sim_distractor_rejection_and_6d_brick_localization():
    pytest.importorskip("mujoco")
    pytest.importorskip("frax")

    from brickcell.block_inspector.real2sim import (
        Real2SimPipeline,
        Real2SimSceneTracker,
        synthesize_cluttered_scene_point_cloud,
    )

    pts_w, cols_w, gt_blocks, meta = synthesize_cluttered_scene_point_cloud(
        num_blocks=4,
        seed=42,
        include_human_arm=True,
        include_wires=True,
        include_trash=True,
        include_robot_arm_points=True,
    )
    pipeline = Real2SimPipeline(enable_temporal_tracking=True)
    state = pipeline.perceive_point_cloud(
        pts_w,
        cols_w,
        q_arm_rad=np.asarray(meta["q_arm_rad"], dtype=np.float64),
        provenance="test_synthetic_clutter",
    )

    # All 4 official DUPLO parts are localized with < 4 mm position error and high ICP fitness
    assert len(state.bricks) == 4
    perceived_names = {b.spec.name for b in state.bricks}
    assert perceived_names == set(DUPLO_CATALOG.keys())
    for b in state.bricks:
        assert b.icp_fitness >= 0.90
        assert b.icp_rmse_m <= 0.005
        gt_match = [g for g in gt_blocks if g.spec.name == b.spec.name]
        assert len(gt_match) == 1
        assert gt_match[0].initial_T_world_block is not None
        pos_err_m = float(
            np.linalg.norm(b.T_world_block[:3, 3] - gt_match[0].initial_T_world_block[:3, 3])
        )
        assert pos_err_m < 0.004

    # All 3 IKEA KALAS plates (source, good, defective) are localized
    assert set(state.plates.keys()) == {"source", "good", "defective"}
    assert float(np.linalg.norm(state.plates["source"].pos_world[:2] - [0.23, 0.0])) < 0.005

    # Human arm, wires, trash, and robot arm link points are 100% rejected
    d_counts = state.distractor_counts_by_category()
    assert d_counts["human_arm"] >= 1
    assert d_counts["wire"] >= 2
    assert d_counts["trash"] >= 3
    assert d_counts["robot_arm"] >= 1

    # Verify temporal scene tracker preserves stable IDs across frames
    tracker = Real2SimSceneTracker(ema_alpha=0.5)
    tracked_1 = tracker.update(state.bricks)
    tracked_2 = tracker.update(state.bricks)
    assert [b.block_id for b in tracked_1] == [f"block_{i}" for i in range(4)]
    assert [b.spec.name for b in tracked_1] == [b.spec.name for b in tracked_2]


def test_real2sim_teammate_depth_packets_and_ply_ingestion(tmp_path: Path):
    import base64
    import io
    import struct

    from PIL import Image

    from brickcell.block_inspector.real2sim import (
        Real2SimPipeline,
        decode_live_depth_snapshot,
        load_point_cloud_file,
        save_point_cloud_ply,
    )

    pipeline = Real2SimPipeline()
    rig = pipeline.rig

    # 1. ASCII PLY round-trip
    pts_orig = np.array([[0.22, 0.01, 0.018], [0.23, -0.01, 0.022]], dtype=np.float64)
    cols_orig = np.array([[204, 20, 20], [12, 84, 204]], dtype=np.uint8)
    ascii_ply = save_point_cloud_ply(tmp_path / "cloud_ascii.ply", pts_orig, cols_orig)
    pts_loaded, cols_loaded = load_point_cloud_file(ascii_ply, rig=rig)
    assert np.allclose(pts_orig, pts_loaded, atol=1e-4)
    assert np.array_equal(cols_orig, cols_loaded)

    # 2. RealSense SDK binary_little_endian PLY in SDK viewer axes (x, -y, -z)
    bin_ply = tmp_path / "350122060653.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    # Point in camera optical frame (0, 0, 0.65) -> SDK viewer axes (0, 0, -0.65)
    body = struct.pack("<fffBBB", 0.0, 0.0, -0.65, 220, 30, 30) + struct.pack(
        "<fffBBB", 0.01, -0.01, -0.65, 30, 180, 70
    )
    bin_ply.write_bytes(header + body)
    pts_bin, cols_bin = load_point_cloud_file(bin_ply, rig=rig)
    assert len(pts_bin) == 2
    assert cols_bin.shape == (2, 3)

    # 3. Teammate `live_depth.py` /snapshot/top base64 JPEG + 16-bit PNG packet
    rgb_img = np.full((60, 80, 3), [210, 25, 25], dtype=np.uint8)
    depth_u16 = np.full((60, 80), 680, dtype=np.uint16)
    buf_rgb = io.BytesIO()
    Image.fromarray(rgb_img).save(buf_rgb, format="JPEG")
    buf_d = io.BytesIO()
    Image.fromarray(depth_u16).save(buf_d, format="PNG")
    snap_pkt = {
        "camera": "top",
        "serial": "350122060653",
        "width": 80,
        "height": 60,
        "depth_scale_m": 0.001,
        "intrinsics": {"fx": 75.0, "fy": 75.0, "ppx": 40.0, "ppy": 30.0},
        "rgb_jpeg_b64": base64.b64encode(buf_rgb.getvalue()).decode("ascii"),
        "depth_png_b64": base64.b64encode(buf_d.getvalue()).decode("ascii"),
    }
    pts_snap, cols_snap = decode_live_depth_snapshot(snap_pkt, rig=rig, stride=4)
    assert len(pts_snap) > 10
    assert cols_snap.shape == (len(pts_snap), 3)

    # 4. Teammate capture directory with `top-points.json`, `plate-points.json`, and `provisional-transform.json`
    cap_dir = tmp_path / "capture_01"
    cap_dir.mkdir(parents=True, exist_ok=True)
    # Transform a dense red 2x2 brick surface from world into top & plate camera frames
    T_wt = rig.realsense_1.world_transform()
    T_tw = np.linalg.inv(T_wt)
    gx, gy = np.meshgrid(np.linspace(0.214, 0.246, 12), np.linspace(-0.016, 0.016, 12))
    brick_w = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.0276)])
    brick_cam = (brick_w @ T_tw[:3, :3].T) + T_tw[:3, 3]
    top_pts_list = [[float(p[0]), float(p[1]), float(p[2]), 204, 20, 20] for p in brick_cam]
    (cap_dir / "top-points.json").write_text(
        json.dumps({"serial": "350122060653", "points": top_pts_list}), encoding="utf-8"
    )
    (cap_dir / "provisional-transform.json").write_text(
        json.dumps(
            {
                "top_from_plate_R": np.eye(3).tolist(),
                "top_from_plate_t_m": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    cap_state = pipeline.perceive_capture_directory(cap_dir)
    assert len(cap_state.bricks) == 1
    assert cap_state.bricks[0].spec.name == "duplo_2x2_3437"


def test_real2sim_closed_loop_replication_and_policy_rollout(tmp_path: Path):
    pytest.importorskip("mujoco")
    pytest.importorskip("frax")

    from brickcell.block_inspector.real2sim import (
        Real2SimPipeline,
        synthesize_cluttered_scene_point_cloud,
    )

    pts_w, cols_w, gt_blocks, meta = synthesize_cluttered_scene_point_cloud(
        num_blocks=1,
        seed=42,
        include_human_arm=True,
        include_wires=True,
        include_trash=True,
        include_robot_arm_points=True,
    )
    pipeline = Real2SimPipeline()
    scene_state = pipeline.perceive_point_cloud(
        pts_w,
        cols_w,
        q_arm_rad=np.asarray(meta["q_arm_rad"], dtype=np.float64),
        provenance="test_closed_loop",
    )
    report = pipeline.close_the_loop(
        scene_state=scene_state,
        controller="chunked_mlp_policy",
        show_viewer=False,
        realtime_pacing=False,
        output_dir=tmp_path / "r2s_out",
        ground_truth_blocks=gt_blocks,
    )
    assert report.execution_succeeded is True
    assert report.replicated_brick_count == 1
    assert report.mean_replication_pos_err_mm < 3.0
    assert report.distractor_counts["human_arm"] >= 1
    assert report.distractor_counts["wire"] >= 2
    assert report.distractor_counts["trash"] >= 3
    assert report.sorted_counts["good"] + report.sorted_counts["defective"] == 1
