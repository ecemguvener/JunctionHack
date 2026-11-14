"""Tests for the station twin perception (brickcell.twin.perception, geometry, synthetic).

- Geometry helpers: pure numpy/OpenCV, run in the core environment.
- Real fixtures: skip if artifacts/twin-fixtures/fixture-00N is not there.
- Synthetic ground truth: MuJoCo renders; skip if MuJoCo is not installed.

Symmetric parts: a 2x2 brick turned by 90 deg about its own Z axis has the same shape. The
rotation error is the smallest angle between the estimate and truth @ S, for S in the
rotation symmetries of the part (found from the mesh: 2x2 -> 4, 2x4 and 1x2x2 -> 2,
2302 -> 1). These rotations are about the mesh Z axis through the mesh origin, so the
position error of the mesh origin does not change.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from brickcell.twin import geometry as geo
from brickcell.twin import perception as P
from brickcell.twin.state import PLATE_NAMES, StationState, quat_to_matrix

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "artifacts" / "twin-fixtures"


# ----------------------------------------------------------------------------- geometry


def test_rodrigues_and_yaw():
    r = geo.rodrigues([0, 0, 0.7])
    assert np.allclose(r, geo.rot_z(0.7))
    assert math.isclose(geo.yaw_of(r), 0.7)
    assert math.isclose(
        geo.rotation_angle_deg(np.eye(3), geo.rodrigues([0.3, 0, 0])), math.degrees(0.3)
    )


def test_plane_ransac_and_correction():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.3, 0.3, (3000, 2))
    tilt = geo.rodrigues([0.02, -0.01, 0])
    pts = np.column_stack((xy, np.zeros(len(xy)))) @ tilt.T + [0, 0, 0.007]
    pts += rng.normal(0, 0.0005, pts.shape)
    outliers = rng.uniform(-0.3, 0.3, (300, 3))
    n, d, inliers = geo.fit_plane_ransac(np.vstack((pts, outliers)), 0.002)
    assert inliers[: len(pts)].mean() > 0.95
    r, t = geo.plane_to_z0(n, d, np.array([0.0, 0.0, 0.0]))
    moved = pts @ r.T + t
    assert np.abs(moved[:, 2]).max() < 0.003


def test_circle_hough_partial_arc_with_profile_radius():
    rng = np.random.default_rng(1)
    centre = np.array([0.31, -0.07])
    ang = rng.uniform(0, math.radians(200), 2000)  # only 200 deg of the wall in view
    z = rng.uniform(0.006, 0.023, len(ang))
    radii = P._plate_wall_radius(z) * 0.97
    xy = centre + np.column_stack((np.cos(ang), np.sin(ang))) * radii[:, None]
    xy += rng.normal(0, 0.001, xy.shape)
    acc, origin = geo.circle_hough(xy, P._plate_wall_radius(z), [0, -0.3], [0.6, 0.3], 0.004)
    iy, ix = np.unravel_index(np.argmax(acc), acc.shape)
    guess = origin + 0.004 * np.array([ix, iy])
    assert np.linalg.norm(guess - centre) < 0.01
    fitted, scale = geo.fit_circle_known_radius(
        xy, P._plate_wall_radius(z), guess, scale_range=(0.9, 1.1)
    )
    assert np.linalg.norm(fitted - centre) < 0.0015
    assert abs(scale - 0.97) < 0.01
    assert 180 <= geo.angular_coverage_deg(xy, fitted) <= 220


def test_deproject_project_round_trip():
    depth = np.full((48, 64), 0.8, np.float32)
    depth[10, 20] = 0.0
    pts, pix = geo.deproject(depth, 60.0, 61.0, 31.5, 23.5)
    assert len(pts) == depth.size - 1
    r = geo.rodrigues([0.2, -0.1, 0.3])
    t = np.array([0.1, 0.2, 0.9])
    uv, z = geo.project(pts @ r.T + t, r, t, 60.0, 61.0, 31.5, 23.5)
    assert np.allclose(z, 0.8)
    assert np.allclose(uv[:, 0], pix % 64) and np.allclose(uv[:, 1], pix // 64)


def test_closest_point_grid_matches_brute_force():
    rng = np.random.default_rng(2)
    samples = rng.uniform(-0.01, 0.01, (400, 3))
    grid = geo.ClosestPointGrid.build(samples, 0.001, 0.005)
    q = rng.uniform(-0.012, 0.012, (200, 3))
    idx = grid.lookup(q)
    brute = np.linalg.norm(q[:, None] - samples[None], axis=2).min(1)
    got = np.linalg.norm(q - samples[idx], axis=1)
    assert np.all(idx >= 0)
    assert np.all(got - brute < 0.0012)  # within one voxel of the exact neighbour
    assert grid.lookup(np.array([[1.0, 0, 0]]))[0] == -1


def test_part_models_symmetries_and_poses():
    models = P.part_models(P.load_perception_config())
    assert {k: len(m.symmetries) for k, m in models.items()} == {
        "part-001": 4,
        "part-002": 2,
        "part-003": 2,
        "part-004": 1,
    }
    for m in models.values():
        assert m.poses, m.object_id
        for pose in m.poses:  # the stored pose puts the long side of the footprint on X
            assert pose.long >= pose.short - 1e-3
    # 2x2: the four "side_*" poses are one shape; only one is kept.
    assert sum(p.pose_id.startswith("side_") for p in models["part-001"].poses) == 1


def test_symmetric_rotation_error():
    syms = [geo.rot_z(k * math.pi / 2) for k in range(4)]
    r = geo.rodrigues([0.3, 0.2, 0.1])
    assert geo.symmetric_rotation_error_deg(r, r @ geo.rot_z(math.pi / 2), syms) < 1e-4
    assert geo.symmetric_rotation_error_deg(r, r @ geo.rot_z(math.pi / 2), syms[:1]) > 89


def test_config_defaults_and_unknown_keys(tmp_path):
    cfg = P.load_perception_config()
    assert set(P.DEFAULT_CONFIG) == set(cfg)
    bad = tmp_path / "twin.yaml"
    bad.write_text("perception:\n  nonsense: 1\n")
    with pytest.raises(ValueError):
        P.load_perception_config(bad)
    empty = tmp_path / "empty.yaml"
    empty.write_text("perception: {}\n")
    assert P.load_perception_config(empty) == P.DEFAULT_CONFIG


# ----------------------------------------------------------------------------- real fixtures


def _real(name: str):
    path = FIXTURES / name
    if not (path / "meta.json").is_file():
        pytest.skip(f"{path} is not there")
    from brickcell.simulation.scene_config import load_scene_config
    from brickcell.twin.state import load_fixture

    return load_fixture(path), load_scene_config()


def _check_common(state: StationState):
    StationState.from_json(state.to_json())  # valid contract
    plates = {p.name: np.array(p.pos[:2]) for p in state.plates}
    assert set(plates) == set(PLATE_NAMES)
    # The blue plate has no configured colour: reported, not a sim plate.
    assert any("colour" in r["reason"] for r in state.diagnostics["plates_rejected"])
    assert state.arm is None
    for b in state.bricks:
        assert b.residual_mm is not None and b.residual_mm < 4.0
        assert 0 <= min(b.color) and max(b.color) <= 1
        assert b.pos[2] < 0.06
    return plates


def test_real_fixture_001():
    frames, scene = _real("fixture-001")
    state = P.locate(frames, scene)
    plates = _check_common(state)
    assert np.linalg.norm(plates["defective"] - [0.322, -0.055]) < 0.015  # pink plate on the board
    assert np.linalg.norm(plates["good"] - [0.408, 0.115]) < 0.015  # green plate
    # The refined side pose (calibrate.py, 2026-11-14) agrees with the overhead camera.
    assert state.cameras["side"]["used"] is True
    offsets = state.diagnostics["camera_agreement"]["cameras"]["side"]["median_offset_mm"]
    assert np.hypot(*offsets) < 10
    table_bricks = np.array(
        [
            [-0.036, -0.222],  # real light-blue brick next to the arm
            [0.067, 0.141],  # real yellow 1x2x2 on the table (seen when the side camera is used)
        ]
    )
    in_plate = 0
    for b in state.bricks:
        xy = np.array(b.pos[:2])
        if b.support.startswith("plate_"):
            assert np.linalg.norm(xy - plates[b.support[6:]]) < 0.1
            in_plate += 1
        else:  # no brick on the arm, the can, the laptop or the box
            assert np.linalg.norm(table_bricks - xy, axis=1).min() < 0.02, b
    assert in_plate >= 1
    pink = [b for b in state.bricks if b.support == "plate_defective"]
    assert any(b.part == "3011" for b in pink)  # the red 2x4 in the pink plate


def test_real_fixture_002():
    frames, scene = _real("fixture-002")
    state = P.locate(frames, scene)
    plates = _check_common(state)
    by_id = {b.object_id: b for b in state.bricks}
    red = by_id["part-002"]  # red 2x4, upright, in the pink plate
    assert red.support == "plate_defective" and red.stable_pose == "upright"
    assert np.linalg.norm(np.array(red.pos[:2]) - [0.337, -0.152]) < 0.015
    # The cream 2x4 in the green plate is the same sim part: it is a duplicate.
    dups = state.diagnostics["duplicates"]
    assert any(
        d["object_id"] == "part-002"
        and np.linalg.norm(np.array(d["pos"][:2]) - plates["good"]) < 0.08
        for d in dups
    )
    for b in state.bricks:  # all bricks are in a plate (no brick on the toy, the arm, the can)
        assert b.support.startswith("plate_"), b


# ----------------------------------------------------------------------------- synthetic


@pytest.fixture(scope="module")
def world():
    pytest.importorskip("mujoco")
    from brickcell.simulation.world import World

    w = World(ROOT / "artifacts" / "twin-test-perception-world")
    yield w
    w.close()


def test_mujoco_depth_is_z_depth(world):
    from brickcell.twin import synthetic

    for camera in ("overhead", "side"):
        r = synthetic.check_depth_convention(world, camera)
        assert r["zdepth_median_err"] < 1e-4
        assert r["ray_median_err"] > 1e-3  # it is NOT the distance along the ray


def test_plate_wall_profile_matches_world():
    world_mod = pytest.importorskip("brickcell.simulation.world")
    prof = np.array(world_mod.PLATE_PROFILE) / 1000
    inner = prof[(prof[:, 1] >= 0.004) & (np.arange(len(prof)) <= 9)]
    outer = prof[(np.arange(len(prof)) >= 10) & (np.arange(len(prof)) <= 14)]
    for z, r in zip(P.PLATE_WALL_Z[1:], P.PLATE_WALL_R[1:]):  # the outer table starts at 6.5 mm
        ri = np.interp(z, inner[:, 1], inner[:, 0])
        ro = np.interp(z, outer[::-1, 1], outer[::-1, 0])
        assert abs((ri + ro) / 2 - r) < 0.0015, (z, r, ri, ro)


@pytest.mark.parametrize("noise", ["none", "d415"])
def test_synthetic_ground_truth(world, noise):
    from brickcell.twin import synthetic

    models = P.part_models(P.load_perception_config())
    found = total = 0
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        bricks, plates, truth = synthetic.random_scene(rng, n_in_plate=seed % 3)
        frames = synthetic.render_scene(
            world, bricks, plates, noise=noise, seed=seed, arm_qpos=synthetic.ARM_AWAY
        )
        state = P.locate(frames, world.scene)
        for p in state.plates:
            assert np.linalg.norm(np.array(p.pos[:2]) - plates[p.name]) < 0.003
        assert {p.name for p in state.plates} == {"good", "defective"}
        result = {b.object_id: b for b in state.bricks}
        assert set(result) <= set(bricks)
        for object_id, (pos, quat) in bricks.items():
            total += 1
            b = result.get(object_id)
            if b is None:
                continue
            found += 1
            # Part identity: the brick at this place has the right part (one body per part).
            pos_err = np.linalg.norm(np.array(b.pos) - pos) * 1000
            rot_err = geo.symmetric_rotation_error_deg(
                quat_to_matrix(quat), quat_to_matrix(b.quat), models[object_id].symmetries
            )
            assert pos_err <= 5.0 and rot_err <= 5.0, (
                seed,
                object_id,
                truth[object_id],
                b,
                pos_err,
                rot_err,
            )
            expected = (
                f"plate_{truth[object_id]['plate']}" if truth[object_id]["plate"] else "table"
            )
            assert b.support == expected, (seed, object_id, b.support, expected)
    assert found >= 0.9 * total, (found, total)
