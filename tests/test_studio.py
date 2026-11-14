"""Tests for Physical AI Studio SO-101 Analytical Pick-and-Place Architecture integration."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from brickcell.studio.engine import (
    MOTOR_NAMES,
    SO101StudioEngine,
)
from brickcell.studio.server import (
    handle_studio_get,
    handle_studio_post,
)


def test_studio_unit_conversions_and_normalized_bridge() -> None:
    """Verify round-trip conversions across normalized [-100, 100], ticks, radians, and degrees."""
    engine = SO101StudioEngine()
    q_rad = np.array([0.12, -0.85, 0.96, 0.34, 0.55], dtype=np.float64)

    # Rad -> Normalized -> Rad roundtrip
    norm = engine.radians_to_normalized(q_rad, arm="follower")
    assert norm.shape == (5,)
    assert np.all(norm >= -100.0) and np.all(norm <= 100.0)

    q_rad_back = engine.normalized_to_radians(norm, arm="follower")
    assert np.allclose(q_rad, q_rad_back, atol=5e-3)

    # Ticks -> Normalized -> Ticks roundtrip
    ticks = engine.follower_cal.radians_to_ticks(q_rad)
    norm_from_ticks = engine.ticks_to_normalized(engine.follower_cal, ticks)
    ticks_back = engine.normalized_to_ticks(engine.follower_cal, norm_from_ticks)
    assert np.all(np.abs(ticks - ticks_back) <= 1)


def test_studio_fk_jacobian_and_5dof_ik() -> None:
    """Verify 4x4 SE(3) FK, 6x5 Jacobians, and 5-DOF damped Gauss-Newton IK convergence."""
    engine = SO101StudioEngine()
    kin = engine.compute_kinematics_and_jacobian([0.0, -0.52, 1.12, 0.82, 0.0], arm="follower")

    assert len(kin["T_base_tcp"]) == 4
    assert len(kin["T_world_tcp"]) == 4
    assert len(kin["jacobian_base_rad"]) == 6
    assert len(kin["jacobian_base_rad"][0]) == 5
    assert len(kin["jacobian_base_tick"]) == 6
    assert len(kin["jacobian_base_lerobot_deg"]) == 6
    assert kin["manipulability_5d"] > 0.0
    assert len(kin["singular_values"]) == 5

    # Solve 5-DOF IK for a reachable point in the workspace
    sol = engine.solve_5dof_ik(
        target_xyz_world=[0.298, -0.076, 0.065],
        pitch_down_deg=90.0,
        jaw_yaw_deg=0.0,
    )
    assert sol["feasible"] is True
    assert sol["position_error_mm"] < 5.0
    assert len(sol["q_enc_rad"]) == 5
    assert len(sol["ticks"]) == 5


def test_studio_published_cad_dynamics() -> None:
    """Verify published CAD mass priors, 5x5 mass matrix symmetry/PSD, and gravity torques."""
    engine = SO101StudioEngine()
    dyn = engine.compute_dynamics([0.0, -0.45, 0.95, 0.50, 0.45], payload_mass_kg=0.019)

    assert abs(dyn["arm_mass_g"] - 632.006) < 0.05
    assert abs(dyn["total_mass_g"] - (632.006 + 19.0)) < 0.05

    M = np.asarray(dyn["mass_matrix_5x5_kg_m2"], dtype=np.float64)
    assert M.shape == (5, 5)
    assert np.allclose(M, M.T, atol=1e-9)
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0.0)

    assert len(dyn["gravity_torque_nm"]) == 5
    assert len(dyn["feedforward_gravity_ticks"]) == 5


def test_studio_leader_to_follower_and_alignment_derivation() -> None:
    """Verify Leader -> Follower CAD posture mapping and reference tick alignment derivation."""
    engine = SO101StudioEngine()
    leader_ticks = [2068, 965, 3052, 2701, 2003]
    mapped = engine.map_leader_to_follower(leader_ticks=leader_ticks, leader_gripper_pct=35.0)
    assert len(mapped["follower_ticks"]) == 5
    assert len(mapped["follower_q_enc_rad"]) == 5
    assert len(mapped["q_cad_rad"]) == 5

    derived = engine.derive_alignment_from_reference_ticks(
        arm="follower",
        measured_ticks=[2019, 882, 3047, 2906, 2070],
    )
    assert derived["max_reconstruction_error_deg"] < 1e-4
    assert len(derived["cad_at_zero_rad"]) == 5


def test_studio_db_sync_and_http_handlers(tmp_path: Path) -> None:
    """Verify SQLite physicalai.db auto-discovery and /api/so101/* HTTP handlers."""
    db_file = tmp_path / "physicalai.db"
    conn = sqlite3.connect(db_file)
    try:
        conn.execute(
            "CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, created_at TEXT, updated_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE project_robots (id TEXT PRIMARY KEY, project_id TEXT, name TEXT, type TEXT, payload TEXT)"
        )
        conn.execute(
            "CREATE TABLE project_cameras (id TEXT PRIMARY KEY, project_id TEXT, name TEXT, driver TEXT, fingerprint TEXT, hardware_name TEXT, payload TEXT)"
        )
        pid = "e97d7168-36bc-400a-9013-fc3c14b823f2"
        conn.execute(
            "INSERT INTO projects VALUES (?, 'white-block-picker', '2026-11-14', '2026-11-14')",
            (pid,),
        )
        engine_default = SO101StudioEngine()
        follower_payload = {
            "serial_number": "5B3D040991",
            "port": "/dev/ttyACM1",
            "unit": "normalized",
            "calibration": engine_default.follower_cal.records,
        }
        conn.execute(
            "INSERT INTO project_robots VALUES ('r1', ?, 'follower', 'SO101_Follower', ?)",
            (pid, json.dumps(follower_payload)),
        )
        conn.execute(
            "INSERT INTO project_cameras VALUES ('c1', ?, 'top', 'realsense', '350122060653', 'Intel RealSense D415', ?)",
            (
                pid,
                json.dumps(
                    {"serial_number": "350122060653", "width": 640, "height": 480, "fps": 30}
                ),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    engine = SO101StudioEngine()
    sync_info = engine.sync_from_studio_db(db_path=db_file)
    assert sync_info["synced"] is True
    assert sync_info["active_project"]["name"] == "white-block-picker"
    assert len(sync_info["robots"]) == 1
    assert len(sync_info["cameras"]) == 1
    assert len(MOTOR_NAMES) == 5

    # HTTP GET /api/so101/state
    get_res = handle_studio_get(engine, "/api/so101/state")
    assert get_res is not None
    status, ctype, body, _headers = get_res
    assert status == 200
    assert "application/json" in ctype
    parsed = json.loads(body.decode("utf-8"))
    assert "kinematics" in parsed and "dynamics" in parsed

    # HTTP GET /api/so101/ui-extension.js
    js_res = handle_studio_get(engine, "/api/so101/ui-extension.js")
    assert js_res is not None
    assert js_res[0] == 200
    assert "application/javascript" in js_res[1]

    # HTTP POST /api/so101/ik
    post_res = handle_studio_post(
        engine,
        "/api/so101/ik",
        {
            "target_xyz_world": [0.298, -0.076, 0.065],
            "pitch_down_deg": 90.0,
            "jaw_yaw_deg": 0.0,
            "update_virtual_arm": True,
        },
    )
    assert post_res is not None
    assert post_res[0] == 200
    ik_payload = post_res[1]
    assert ik_payload["feasible"] is True
    assert len(ik_payload["normalized"]) == 5
