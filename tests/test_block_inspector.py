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


def test_real2sim_live_mirror_dynamic_plates_bricks_and_arm_pickup():
    pytest.importorskip("mujoco")
    pytest.importorskip("frax")

    from brickcell.block_inspector.geometry import DUPLO_CATALOG
    from brickcell.block_inspector.real2sim import (
        Real2SimPipeline,
        synthesize_cluttered_scene_point_cloud,
    )
    from brickcell.block_inspector.twin import SO101InspectionTwin, SpawnedBlockInfo

    pts_w, cols_w, _, meta = synthesize_cluttered_scene_point_cloud(
        num_blocks=3,
        seed=42,
        include_human_arm=True,
        include_wires=True,
        include_trash=True,
        include_robot_arm_points=True,
    )
    pipeline = Real2SimPipeline(enable_temporal_tracking=True)
    scene_state = pipeline.perceive_point_cloud(
        pts_w,
        cols_w,
        q_arm_rad=np.asarray(meta["q_arm_rad"], dtype=np.float64),
        provenance="test_dynamic_mirror",
    )
    assert len(scene_state.bricks) == 3

    slot_specs = list(DUPLO_CATALOG.values())
    prealloc = [
        SpawnedBlockInfo(
            block_id=f"block_{i}",
            spec=slot_specs[i % len(slot_specs)],
            initial_xy_yaw=(0.0, 0.0, 0.0),
        )
        for i in range(6)
    ]
    twin = SO101InspectionTwin(
        show_viewer=False,
        realtime_pacing=False,
        spawned_blocks=prealloc,
        plate_positions=scene_state.plate_positions_dict(),
    )
    try:
        # 1. Mirror live arm joint angles + gripper state directly
        q_target = np.array([-0.35, -0.55, 1.15, -0.35, 0.40], dtype=np.float64)
        twin.set_arm_state_direct(q_target, gripper_pct=0.25, sync_viewer=False)
        assert np.allclose(twin.current_arm_q(), q_target, atol=1e-4)

        # 2. Simulate moving the source plate in reality and syncing into MuJoCo
        import mujoco

        scene_state.plates["source"].pos_world[:] = [0.295, -0.072, 0.0]
        twin.sync_real2sim_scene(scene_state, settle_steps=0)
        src_body_id = mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_BODY, "plate_source")
        assert np.allclose(twin.data.xpos[src_body_id][:2], [0.295, -0.072], atol=1e-4)

        # Unused preallocated slots (block_3..block_5) are hidden beneath the table (z < -0.2 m)
        unused_body_id = twin.block_body_ids["block_5"]
        assert float(twin.data.xpos[unused_body_id][2]) < -0.20

        # 3. Simulate picking up block_0 with the gripper and lifting it to z = 0.14 m
        assert pipeline.tracker is not None
        T_ee_near = scene_state.bricks[0].T_world_block.copy()
        pipeline.tracker.update_latched_brick_from_arm(T_ee_near, gripper_pct=0.35)
        T_ee_lifted = T_ee_near.copy()
        T_ee_lifted[2, 3] = 0.140
        lifted_bricks = pipeline.tracker.update_latched_brick_from_arm(
            T_ee_lifted, gripper_pct=0.35
        )
        scene_state.bricks = lifted_bricks
        twin.sync_real2sim_scene(scene_state, settle_steps=0)
        b0_body_id = twin.block_body_ids["block_0"]
        assert abs(float(twin.data.xpos[b0_body_id][2]) - 0.140) < 0.005
    finally:
        twin.close()


def test_physical_cell_dark_mark_inspector_dashboard_and_executor(monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("frax")
    import time

    from brickcell.block_inspector.camera_adapters import TwinDashboardReporter
    from brickcell.block_inspector.hardware_cell import (
        LiveCameraSnapshot,
        PhysicalArmController,
        PhysicalSO101CellExecutor,
        RealCameraAnomalyInspector,
        StaleCameraFrameError,
        score_brick_roi_dark_mark,
    )
    from brickcell.block_inspector.real2sim import (
        PerceivedBrick,
        PerceivedPlate,
        Real2SimSceneState,
        project_world_obb_to_image_box,
        real_intel_camera_poses_world,
    )

    # 1. Symmetric-flanked CIE-Lab dark-mark scorer: 0.0 on clean brick, >16.0 on marked brick
    clean_roi = np.full((64, 64, 3), [212, 174, 58], dtype=np.uint8)
    rng = np.random.default_rng(11)
    clean_roi = np.clip(clean_roi + rng.integers(-6, 7, size=clean_roi.shape), 0, 255).astype(
        np.uint8
    )
    clean_score, clean_dets = score_brick_roi_dark_mark(clean_roi)
    assert clean_score == 0.0 and clean_dets == []

    marked_roi = clean_roi.copy()
    cv2.line(marked_roi, (18, 30), (46, 32), (24, 22, 20), thickness=2)
    mark_score, mark_dets = score_brick_roi_dark_mark(marked_roi)
    assert mark_score >= 16.0
    assert len(mark_dets) >= 1 and mark_dets[0]["label"] == "dark mark"

    # 2. RealCameraAnomalyInspector: zero robot handles, calibrated threshold required,
    #    and timed-out worker permanently retired (`AGENTS.md`)
    with pytest.raises(ValueError, match="dark_mark_threshold"):
        RealCameraAnomalyInspector(dark_mark_threshold=None)

    inspector = RealCameraAnomalyInspector(dark_mark_threshold=16.0, worker_timeout_s=0.05)
    assert not hasattr(inspector, "robot") and not hasattr(inspector, "arm")
    full_img = np.zeros((480, 640, 3), dtype=np.uint8)
    full_img[180:244, 260:324] = marked_roi
    full_dep = np.full((480, 640), 0.30, dtype=np.float32)
    res_ok = inspector.evaluate_frame(
        full_img, full_dep, [260, 180, 324, 244], expected_z_m=0.30, z_band_m=0.05
    )
    assert res_ok.valid_view and res_ok.flagged and res_ok.raw_anomaly_distance >= 16.0
    inspector.close()

    slow_inspector = RealCameraAnomalyInspector(dark_mark_threshold=16.0, worker_timeout_s=0.03)
    orig_score = slow_inspector._score_sync
    slow_inspector._score_sync = lambda *a, **kw: (time.sleep(0.18), orig_score(*a, **kw))[1]  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="retired"):
        slow_inspector.evaluate_frame(full_img, full_dep, [260, 180, 324, 244], expected_z_m=0.30)
    assert slow_inspector.timed_out is True
    with pytest.raises(TimeoutError, match="cannot be reused"):
        slow_inspector.evaluate_frame(full_img, full_dep, [260, 180, 324, 244], expected_z_m=0.30)

    # 3. TwinDashboardReporter (`/api/focus` + `/api/anomalies` schema for `:8096`)
    posted: list[tuple[str, dict]] = []
    dash = TwinDashboardReporter(base_url="http://127.0.0.1:8096")
    monkeypatch.setattr(
        dash, "_post_json", lambda path, payload: posted.append((path, payload)) or True
    )
    assert dash.report_focus("part-003", "duplo_1x2x2_76371 (target)", [280, 210, 330, 255])
    assert dash.report_anomalies(
        [
            {
                "box": [288, 222, 316, 238],
                "score": 29.75,
                "label": "dark mark",
                "object_id": "part-003",
            }
        ]
    )
    assert posted[0][0] == "/api/focus"
    assert posted[0][1]["object_id"] == "part-003" and posted[0][1]["box"] == [280, 210, 330, 255]
    assert posted[1][0] == "/api/anomalies"
    assert posted[1][1]["items"][0]["label"] == "dark mark"

    # 4. PhysicalSO101CellExecutor + CogManInspectionAgent closed-loop dry-run & stale-frame guard
    spec = DUPLO_CATALOG["duplo_2x2_3437"]
    T_wb = np.eye(4, dtype=np.float64)
    T_wb[:3, 3] = [0.275, -0.015, 0.016]
    T_mesh = T_wb.copy()
    T_mesh[2, 3] = 0.0
    brick = PerceivedBrick(
        block_id="block_0",
        spec=spec,
        T_world_block=T_wb,
        T_world_mesh_origin=T_mesh,
        stable_pose_name="pose_0",
        xy_yaw=(0.275, -0.015, 0.0),
        support_region="source",
        icp_fitness=0.95,
        icp_rmse_m=0.002,
        confidence=0.95,
        has_visual_defect=True,
        num_points=120,
        mean_rgb=(210.0, 170.0, 55.0),
        cluster_pts_world=np.tile([0.275, -0.015, 0.016], (20, 1)),
    )
    plates = {
        "source": PerceivedPlate(
            "source", np.array([0.288, -0.019, 0.0]), 0.095, 400, (0, 140, 180)
        ),
        "good": PerceivedPlate("good", np.array([0.374, 0.150, 0.0]), 0.095, 400, (90, 210, 110)),
        "defective": PerceivedPlate(
            "defective", np.array([0.215, 0.144, 0.0]), 0.095, 400, (220, 120, 150)
        ),
    }

    R_pl, p_pl, K_pl = real_intel_camera_poses_world()["plate"]
    p_box, z_pl = project_world_obb_to_image_box(
        T_wb, spec.dims_m, R_pl, p_pl, K_pl, width=640, height=480, pad_px=22
    )
    assert p_box is not None
    plate_rgb = np.full((480, 640, 3), [212, 174, 58], dtype=np.uint8)
    plate_rgb = np.clip(plate_rgb + rng.integers(-5, 6, size=plate_rgb.shape), 0, 255).astype(
        np.uint8
    )
    mx = (p_box[0] + p_box[2]) // 2
    my = (p_box[1] + p_box[3]) // 2
    cv2.line(plate_rgb, (mx - 12, my), (mx + 12, my + 1), (20, 20, 20), thickness=2)
    plate_dep = np.full((480, 640), float(z_pl), dtype=np.float32)

    class _MockBenchSource:
        live_depth_url = "http://127.0.0.1:8094"

        def __init__(self) -> None:
            self.calls = 0
            self.fail_stale = False

        def fetch_d415_snapshot(self, view: str = "plate") -> LiveCameraSnapshot:
            if self.fail_stale:
                raise StaleCameraFrameError("mock stale frame")
            self.calls += 1
            return LiveCameraSnapshot(
                camera_name=view,
                rgb=plate_rgb,
                depth_m=plate_dep,
                timestamp_s=time.time(),
                age_s=0.02,
                frame_id=f"{view}-{self.calls}",
            )

        def fetch_wrist_snapshot(self, resize_hw=(360, 640)) -> LiveCameraSnapshot:
            return LiveCameraSnapshot(
                camera_name="wrist",
                rgb=np.full((360, 640, 3), 120, dtype=np.uint8),
                depth_m=None,
                timestamp_s=time.time(),
                age_s=0.01,
                frame_id=f"wrist-{self.calls}",
            )

        def perceive_scene(self, _r2s, q_arm_rad=None) -> Real2SimSceneState:
            return Real2SimSceneState(
                timestamp_s=time.time(),
                frame_index=1,
                bricks=[brick],
                plates=plates,
                rejected_clusters=[],
                raw_point_count=1000,
                downsampled_point_count=1000,
                robot_masked_point_count=900,
                foreground_point_count=200,
                latency_ms=4.5,
                provenance="mock_live_bench",
            )

    mock_src = _MockBenchSource()
    arm_ctrl = PhysicalArmController(
        dry_run=True,
        bridge=ArmdFollowerBridge(request_fn=lambda *_a, **_k: {"joints": {}}),
    )
    live_insp = RealCameraAnomalyInspector(dark_mark_threshold=16.0)
    exec_cell = PhysicalSO101CellExecutor(
        frame_source=mock_src,  # type: ignore[arg-type]
        arm=arm_ctrl,
        inspector=live_insp,
        dashboard=dash,
        max_cycles=1,
    )
    try:
        agent = CogManInspectionAgent(executor=exec_cell, ledger=exec_cell.ledger)
        rep = agent.run(instruction="clear source plate", mode="clear_source_plate")
        # The uncalibrated Lab fallback cannot authorize either sort pile.
        assert rep.succeeded is False
        assert exec_cell.sorted_counts == {"good": 0, "defective": 0}
        assert exec_cell.batch_sorted_records == []

        # Verify stale camera frame during scan aborts safely without sorting (`AGENTS.md`)
        exec_cell._cycles_completed = 0
        exec_cell.source_plate_cleared = False
        exec_cell.sorted_counts = {"good": 0, "defective": 0}
        assert exec_cell._skill_localize_next_cluster_block().succeeded
        assert exec_cell._skill_pick_block().succeeded
        mock_src.fail_stale = True
        # This subcase tests transport failure before inference, so mock only
        # model availability to get past the new two-camera preflight.
        live_insp.has_detector_for_camera = lambda camera: camera in ("plate", "wrist")
        scan_out = exec_cell._skill_present_and_spin_at_plate_camera()
        assert not scan_out.succeeded and "stale_camera_frame" in scan_out.reason
        sort_out = exec_cell._skill_sort_block_to_pile("good")
        assert not sort_out.succeeded
        assert exec_cell.sorted_counts == {"good": 0, "defective": 0}
    finally:
        exec_cell.close()


def test_anomalib_multicam_rgb_localization_and_movable_plates() -> None:
    """Verify multi-camera Anomalib wrapper, clamping-overload tolerance, and movable-plate 2D RGB."""
    pytest.importorskip("jax")
    pytest.importorskip("frax")
    import cv2

    from brickcell.block_inspector.armd_bridge import ArmdFollowerBridge, ArmFaultError
    from brickcell.block_inspector.frax_ik import SO101FraxIK
    from brickcell.block_inspector.geometry import station_camera_rig
    from brickcell.block_inspector.hardware_cell import (
        RealCameraAnomalyInspector,
        detect_wrist_jaw_corridor,
        localize_plates_and_bricks_top_rgb,
        refine_grasp_with_wrist_rgb,
        refine_plate_drop_with_wrist_rgb,
        servo_grasp_with_jaw_corridor,
        solve_adaptive_grasp_pose,
    )
    from brickcell.block_inspector.real2sim import real_intel_camera_poses_world
    from brickcell.detector.runtime import Prediction
    from brickcell.inspection.threshold import Threshold

    # 1. Gripper clamping overload while holding a rigid brick must NOT trigger emergency stop,
    #    whereas any arm-joint overload MUST still trigger emergency_stop_hold().
    emergency_calls: list[str] = []

    def _mock_req(method: str, path: str, payload: dict | None = None) -> dict:
        if path == "/arms/follower/stop" and method == "POST":
            emergency_calls.append("stop")
        return {"ok": True}

    bridge = ArmdFollowerBridge(request_fn=_mock_req)
    joint_names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    grip_overload_msg = {
        "timestamp": "2026-11-14T18:00:00Z",
        "joints": {
            name: {
                "position": 1490 if name == "gripper" else 2048,
                "online": True,
                "healthy": (name != "gripper"),
                "errors": ["overload"] if name == "gripper" else [],
            }
            for name in joint_names
        },
    }
    snap_ok = bridge.decode_state_message(grip_overload_msg, allow_gripper_overload=True)
    assert snap_ok is not None and snap_ok.gripper_ticks == 1490
    assert emergency_calls == []

    arm_overload_msg = {
        "timestamp": "2026-11-14T18:00:01Z",
        "joints": {
            name: {
                "position": 2048,
                "online": True,
                "healthy": (name != "elbow_flex"),
                "errors": ["overload"] if name == "elbow_flex" else [],
            }
            for name in joint_names
        },
    }
    with pytest.raises(ArmFaultError, match="elbow_flex"):
        bridge.decode_state_message(arm_overload_msg, allow_gripper_overload=True)
    assert emergency_calls == ["stop"]

    # 2. Multi-camera Anomalib wrapper replaces generic Lab heuristic across top, plate, and wrist.
    thr_calib = Threshold(
        value=0.50,
        uncertainty_margin=0.04,
        version="thr-test-v1",
        calibration_manifest_sha256="a" * 64,
        sample_count=40,
    )

    class _FakeAnomalibDetector:
        def __init__(self, score: float, hotspot: bool = False) -> None:
            self.threshold = thr_calib
            self._score = score
            self._hotspot = hotspot
            self.calls = 0

        def predict(self, rgb: np.ndarray) -> Prediction:
            self.calls += 1
            amap = np.full(rgb.shape[:2], 0.15, dtype=np.float32)
            if self._hotspot:
                h, w = amap.shape
                amap[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 0.92
            return Prediction(score=self._score, latency_ms=3.2, anomaly_map=amap)

    top_det = _FakeAnomalibDetector(0.21)
    plate_det = _FakeAnomalibDetector(0.28)
    wrist_det = _FakeAnomalibDetector(0.84, hotspot=True)
    reinspect_det = _FakeAnomalibDetector(0.49)  # within 0.04 uncertainty_margin of 0.50!

    # Natural textured clean brick image (no dark mark for heuristic), yet wrist Anomalib flags defect!
    rng = np.random.default_rng(7)
    clean_rgb = np.clip(
        np.full((360, 640, 3), [210, 175, 55], dtype=np.int16)
        + rng.integers(-12, 13, size=(360, 640, 3)),
        0,
        255,
    ).astype(np.uint8)
    anomalib_insp = RealCameraAnomalyInspector(
        dark_mark_threshold=16.0,
        camera_detectors={
            "top": {"rgb": top_det},
            "plate": {"rgb": plate_det},
            "wrist": {"rgb": wrist_det},
        },
    )
    try:
        assert anomalib_insp.has_anomalib_models is True
        assert anomalib_insp.inspector_mode == "anomalib_openvino_cpu"
        assert anomalib_insp.configured_cameras() == ["plate", "wrist"]

        r_top = anomalib_insp.evaluate_frame(
            clean_rgb, None, [100, 80, 260, 220], camera_name="top"
        )
        assert not r_top.valid_view and not r_top.flagged
        assert r_top.reason == "camera_excluded_from_mark_inspection"

        r_wrist = anomalib_insp.evaluate_frame(
            clean_rgb, None, [120, 90, 320, 250], camera_name="wrist"
        )
        assert r_wrist.valid_view and r_wrist.flagged
        assert len(r_wrist.detections) >= 1
        assert "anomalib wrist rgb" in r_wrist.detections[0]["label"]
    finally:
        anomalib_insp.close()

    # Decision.REINSPECT must NOT produce a valid trusted view (`AGENTS.md` safety rule)
    re_insp = RealCameraAnomalyInspector(
        dark_mark_threshold=16.0,
        camera_detectors={"plate": {"rgb": reinspect_det}},
    )
    try:
        r_re = re_insp.evaluate_frame(clean_rgb, None, [100, 80, 260, 220], camera_name="plate")
        assert r_re.valid_view is False
        assert r_re.flagged is False
        assert "reinspect" in r_re.reason
    finally:
        re_insp.close()

    # 3. Depth-free 2D RGB localization tracks shifted/movable KALAS plates + bricks on source plate!
    R_top, p_top, K_top = real_intel_camera_poses_world()["top"]
    top_img = np.full((480, 640, 3), [128, 128, 126], dtype=np.uint8)  # neutral gray table

    def _project_px(xyz_w: list[float]) -> tuple[int, int]:
        pw = np.asarray(xyz_w, dtype=np.float64)
        pc = R_top.T @ (pw - p_top)
        u = int(round(K_top[0, 0] * (pc[0] / pc[2]) + K_top[0, 2]))
        v = int(round(K_top[1, 1] * (pc[1] / pc[2]) + K_top[1, 2]))
        return u, v

    # Shift green and pink plates away from their nominal positions to prove live XY tracking!
    shifted_source = [0.295, -0.028, 0.004]
    shifted_good = [0.352, 0.132, 0.004]
    shifted_def = [0.228, 0.158, 0.004]
    cv2.circle(top_img, _project_px(shifted_source), 58, (25, 155, 195), -1)  # aquamarine RGB
    cv2.circle(top_img, _project_px(shifted_good), 56, (85, 205, 105), -1)  # green RGB
    cv2.circle(top_img, _project_px(shifted_def), 56, (225, 105, 150), -1)  # pink RGB

    # Draw two distinct DUPLO bricks (yellow + red) inside the aquamarine source plate
    b1_uv = _project_px([0.272, -0.052, 0.016])
    b2_uv = _project_px([0.320, 0.002, 0.016])
    cv2.circle(top_img, _project_px(shifted_source), 72, (25, 155, 195), -1)  # full 19 cm plate
    cv2.rectangle(
        top_img, (b1_uv[0] - 11, b1_uv[1] - 11), (b1_uv[0] + 11, b1_uv[1] + 11), (235, 195, 40), -1
    )
    cv2.rectangle(
        top_img, (b2_uv[0] - 16, b2_uv[1] - 9), (b2_uv[0] + 16, b2_uv[1] + 9), (220, 45, 45), -1
    )

    plates_2d, bricks_2d = localize_plates_and_bricks_top_rgb(top_img, R_top, p_top, K_top)
    assert set(plates_2d.keys()) == {"source", "good", "defective"}
    assert np.linalg.norm(plates_2d["good"][:2] - shifted_good[:2]) < 0.025
    assert np.linalg.norm(plates_2d["defective"][:2] - shifted_def[:2]) < 0.025
    assert len(bricks_2d) == 2

    # 4. Eye-in-hand Wrist RGB visual servoing (pick + movable plate drop)
    ik = SO101FraxIK()
    rig = station_camera_rig()
    pre_sol, _ = solve_adaptive_grasp_pose(
        ik,
        np.array([0.282, -0.042, 0.105], dtype=np.float64),
        jaw_yaw_world=0.2,
    )
    wrist_pick_img = np.full((360, 640, 3), [25, 155, 195], dtype=np.uint8)  # aquamarine background
    cv2.rectangle(
        wrist_pick_img, (285, 145), (365, 215), (235, 195, 40), -1
    )  # yellow brick near center
    refined_p, refined_yaw, w_box = refine_grasp_with_wrist_rgb(
        wrist_pick_img,
        q_pre=pre_sol.q_grasp,
        p_tcp_nominal=np.array([0.282, -0.042, 0.026], dtype=np.float64),
        jaw_yaw_nominal=0.2,
        ik=ik,
        wrist_cam=rig.wrist,
    )
    assert w_box is not None and len(w_box) == 4
    assert np.linalg.norm(refined_p[:2] - np.array([0.282, -0.042])) <= 0.030
    assert math.isfinite(refined_yaw)

    drop_sol, _ = solve_adaptive_grasp_pose(
        ik,
        np.array([0.352, 0.132, 0.135], dtype=np.float64),
        jaw_yaw_world=1.9,
        pitches_deg=(72.0, 62.0, 52.0, 42.0, 34.0),
    )
    wrist_drop_img = np.full((360, 640, 3), [128, 128, 126], dtype=np.uint8)
    cv2.circle(wrist_drop_img, (335, 185), 95, (85, 205, 105), -1)  # green bowl slightly offset
    refined_drop_xy = refine_plate_drop_with_wrist_rgb(
        wrist_drop_img,
        pile="good",
        q_above=drop_sol.q_grasp,
        target_xy_nominal=np.array([0.352, 0.132], dtype=np.float64),
        slot_offset_xy=(0.0, 0.0),
        ik=ik,
        wrist_cam=rig.wrist,
    )
    assert np.linalg.norm(refined_drop_xy - np.array([0.352, 0.132])) <= 0.045

    # 5. Inner-jaw purple gripper corridor tracker + specular-glare-immune brick servoing
    jaw_servo_img = np.full((360, 640, 3), [135, 95, 55], dtype=np.uint8)  # wood table
    cv2.circle(jaw_servo_img, (324, 175), 118, (65, 190, 238), -1)  # glossy aquamarine plate
    # Bright ceiling specular glare streak (V=252, S=0) that must NOT merge plate into brick!
    cv2.line(jaw_servo_img, (370, 110), (392, 165), (252, 252, 252), thickness=6)
    # Off-center white 2x4 brick (`RGB=[200, 205, 198]`) sitting near left jaw tip
    cv2.rectangle(jaw_servo_img, (268, 198), (336, 232), (200, 205, 198), -1)
    # Two bottom-anchored royal-purple gripper jaws (`u_left_pad=284`, `u_right_pad=364` -> `u_pinch=324`)
    cv2.fillPoly(
        jaw_servo_img,
        [np.array([[185, 359], [284, 359], [284, 240], [262, 240]], dtype=np.int32)],
        (48, 36, 108),
    )
    cv2.fillPoly(
        jaw_servo_img,
        [np.array([[364, 240], [386, 240], [455, 359], [364, 359]], dtype=np.int32)],
        (48, 36, 108),
    )
    corridor = detect_wrist_jaw_corridor(jaw_servo_img)
    assert corridor is not None
    assert abs(corridor.u_left_pad - 284.0) <= 3.0
    assert abs(corridor.u_right_pad - 364.0) <= 3.0
    assert abs(corridor.u_pinch - 324.0) <= 2.0
    assert abs(corridor.gap_px - 80.0) <= 4.0

    p_srv, yaw_srv, srv_box, straddle_ok, srv_info = servo_grasp_with_jaw_corridor(
        jaw_servo_img,
        q_pose=pre_sol.q_grasp,
        p_tcp_nominal=np.array([0.282, -0.042, 0.021], dtype=np.float64),
        jaw_yaw_nominal=0.2,
        ik=ik,
        wrist_cam=rig.wrist,
    )
    assert srv_info["status"] == "ok"
    assert srv_box is not None and (srv_box[2] - srv_box[0]) < 140  # brick only, not whole plate!
    assert straddle_ok is False  # off-center brick correctly triggers trim before plunge
    assert np.linalg.norm(p_srv[:2] - np.array([0.282, -0.042])) > 0.004
    assert math.isfinite(yaw_srv)
