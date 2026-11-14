"""Tests for brickcell.twin.calibrate: synthetic planes and boards, YAML write, real fixture."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from brickcell.simulation.scene_config import load_scene_config
from brickcell.twin import calibrate as C

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "artifacts/twin-fixtures/fixture-001"
BOARD_IMAGES = {
    "overhead": REPO / "artifacts/twin-dev/calibration/inputs/board-0525-overhead.jpg",
    "side": REPO / "artifacts/twin-dev/calibration/inputs/board-0525-side.jpg",
}
INTR = (600.0, 600.0, 319.5, 239.5)


def look_at(position, target, up=(0, 0, 1)) -> C.Pose:
    """Camera-to-world pose (OpenCV frame) of a camera at ``position`` that looks at target."""
    z = np.asarray(target, float) - position
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return C.Pose(np.column_stack((x, y, z)), np.asarray(position, float))


def perturb(pose: C.Pose, angle_deg: float, axis, shift) -> C.Pose:
    r = C.rot_axis_angle(axis, math.radians(angle_deg))
    return C.Pose(r @ pose.rotation, pose.position + np.asarray(shift, float))


# Uneven grid: line positions (m) of the board, in the board frame.
COLS = np.array([0.0, 0.019, 0.042, 0.067, 0.098, 0.142, 0.190])
ROWS = np.array([0.0, 0.021, 0.038, 0.075, 0.098, 0.146, 0.206, 0.277])
BOARD_YAW = math.radians(20)
BOARD_ORIGIN = np.array([0.20, -0.10])
BOARD_HEIGHT = 0.003


def board_centre() -> np.ndarray:
    return np.array([*board_to_world([[COLS[-1] / 2, ROWS[-1] / 2]])[0], 0.0])


def board_to_world(xy) -> np.ndarray:
    return np.asarray(xy) @ C.rot_z(BOARD_YAW)[:2, :2].T + BOARD_ORIGIN


def render_board(pose: C.Pose, intr=INTR, size=(640, 480), seed=0) -> np.ndarray:
    """RGB image of the orange/tan board (uneven cells) on a wooden table."""
    w, h = size
    v, u = np.mgrid[0:h, 0:w]
    pixels = np.column_stack((u.ravel(), v.ravel())).astype(float)
    ray = pose.rotation @ C.rays(pixels, *intr).T
    scale = (BOARD_HEIGHT - pose.position[2]) / ray[2]
    world = pose.position[:, None] + ray * scale
    local = (world[:2].T - BOARD_ORIGIN) @ C.rot_z(BOARD_YAW)[:2, :2]
    i = np.searchsorted(COLS, local[:, 0]) - 1
    j = np.searchsorted(ROWS, local[:, 1]) - 1
    inside = (i >= 0) & (i < len(COLS) - 1) & (j >= 0) & (j < len(ROWS) - 1)
    image = np.empty((len(pixels), 3), float)
    image[:] = (150, 150, 150)  # grey tape and table
    orange = inside & ((i + j) % 2 == 0)
    image[inside] = (190, 150, 110)  # tan
    image[orange] = (205, 95, 40)  # orange
    image = image.reshape(h, w, 3)
    import cv2

    image = cv2.GaussianBlur(image, (0, 0), 0.8)
    image += np.random.default_rng(seed).normal(0, 2.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def true_corners() -> np.ndarray:
    gx, gy = np.meshgrid(COLS[1:-1], ROWS[1:-1], indexing="ij")
    xy = board_to_world(np.column_stack((gx.ravel(), gy.ravel())))
    return np.column_stack((xy, np.full(len(xy), BOARD_HEIGHT)))


def board_outline(pose: C.Pose) -> np.ndarray:
    """Pixel polygon of the board pattern (like the board_roi setting)."""
    xy = board_to_world(
        [[COLS[0], ROWS[0]], [COLS[-1], ROWS[0]], [COLS[-1], ROWS[-1]], [COLS[0], ROWS[-1]]]
    )
    return pose.project(np.column_stack((xy, np.full(4, BOARD_HEIGHT))), INTR)


def board_plane(pose: C.Pose) -> C.Plane:
    """Camera-frame plane of z = BOARD_HEIGHT."""
    normal = pose.rotation.T @ np.array([0.0, 0.0, 1.0])
    offset = float(normal @ (pose.rotation.T @ (np.array([0, 0, BOARD_HEIGHT]) - pose.position)))
    return C.Plane(normal, -offset)


# ------------------------------------------------------------------------------- planes


def test_plane_fit_and_levelling_recover_the_table():
    rng = np.random.default_rng(1)
    true = look_at(np.array([0.6, 0.3, 0.75]), np.array([0.25, 0.0, 0.0]))
    # Table points (z = 0) and clutter above the table, seen in the camera frame.
    xy = rng.uniform([-0.1, -0.4], [0.6, 0.4], size=(20000, 2))
    table = np.column_stack((xy, rng.normal(0, 0.0015, len(xy))))
    clutter = np.column_stack((rng.uniform(0, 0.4, (3000, 2)), rng.uniform(0.01, 0.1, 3000)))
    points = true.to_camera(np.vstack((table, clutter)))
    plane = C.fit_plane(points, 0.004)
    # The present pose has a 3 deg tilt and a 20 mm height error.
    wrong = perturb(true, 3.0, (1, 0.3, 0), (0.01, -0.005, 0.02))
    before = C.plane_error(plane, wrong)
    assert before["tilt_deg"] > 2.5
    fixed = C.level_pose(wrong, plane, pivot=(0.3, 0.0))
    after = C.plane_error(plane, fixed, (0.3, 0.0))
    assert after["tilt_deg"] < 0.05
    assert abs(after["height_mm"]) < 0.5
    # Levelling keeps x, y of the pivot point on the table.
    anchor_before = _plane_point_below(plane, wrong, (0.3, 0.0))
    anchor_after = _plane_point_below(plane, fixed, (0.3, 0.0))
    assert np.allclose(anchor_after[:2], anchor_before[:2], atol=1e-9)
    assert abs(anchor_after[2]) < 1e-9


def _plane_point_below(plane, pose, xy):
    normal, offset = C.plane_in_world(plane, pose)
    z = -(offset + normal[0] * xy[0] + normal[1] * xy[1]) / normal[2]
    return np.array([xy[0], xy[1], z])


def test_rigid_fits_recover_known_motion():
    rng = np.random.default_rng(2)
    src = rng.uniform(-0.1, 0.1, (30, 3))
    yaw, shift = math.radians(4.0), np.array([0.03, -0.02])
    dst = src @ C.rot_z(yaw).T + [*shift, 0]
    got_yaw, got_shift = C.rigid_fit_2d(src, dst)
    assert abs(got_yaw - yaw) < 1e-9 and np.allclose(got_shift, shift)
    r, t = C.rigid_fit_3d(src, dst)
    assert np.allclose(r, C.rot_z(yaw), atol=1e-9) and np.allclose(t, [*shift, 0], atol=1e-9)


# -------------------------------------------------------------------------------- board


def test_detect_board_finds_uneven_grid_corners():
    true = look_at(np.array([0.50, 0.25, 0.45]), board_centre())
    rgb = render_board(true)
    # The detector gets a pose with a small x, y, yaw error (as after levelling).
    guess = true.turned(math.radians(1.5), (0.008, -0.006))
    det = C.detect_board(rgb, INTR, guess, board_plane(guess), roi_override=board_outline(true))
    expect = true.project(true_corners(), INTR)
    assert len(det.pixels) >= 0.8 * len(expect)
    errors = np.min(np.linalg.norm(det.pixels[:, None] - expect[None], axis=2), axis=1)
    assert np.median(errors) < 0.35
    assert errors.max() < 1.0
    # All inner lines are found. The border against grey tape can also pass the line check
    # (grey and tan have a similar a*), but it gives no corners (the checker test fails).
    assert sorted(len(f) for f in det.lines)[0] >= 5


def test_two_camera_board_fit_recovers_the_side_pose():
    """Grid-index matching and the 2D fit, with projected true corners and pixel noise.

    The side camera sees only part of the board, and its grid numbers start at another line.
    """
    rng = np.random.default_rng(5)
    over = look_at(np.array([0.50, 0.25, 0.45]), board_centre())
    side = look_at(np.array([0.28, -0.38, 0.30]), board_centre())
    side_guess = side.turned(math.radians(-3.0), (0.03, 0.02))  # the present (wrong) pose
    corners = true_corners()
    n_cols, n_rows = len(COLS) - 2, len(ROWS) - 2
    index = np.array([(i, j) for i in range(n_cols) for j in range(n_rows)])
    dets, worlds = {}, {}
    for name, pose, real, keep, offset in (
        ("overhead", over, over, np.ones(len(index), bool), (0, 0)),
        ("side", side_guess, side, index[:, 1] >= 2, (1, -2)),
    ):
        pixels = real.project(corners[keep], INTR) + rng.normal(0, 0.3, (keep.sum(), 2))
        det = C.BoardDetection(
            name, pixels, index[keep] + offset, [[], []], (0.0, 1.0), np.zeros((480, 640), bool)
        )
        dets[name] = det
        worlds[name] = C.corner_world(det, INTR, pose, board_plane(pose))
    pairs, shift, rms, second = C.match_corners(
        dets["overhead"], dets["side"], worlds["overhead"], worlds["side"]
    )
    assert shift == (-1, 2)  # index in overhead = index in side + shift
    assert len(pairs) == int((index[:, 1] >= 2).sum())
    assert rms < 1.0 and second > 3 * rms
    ia, ib = np.array(pairs).T
    yaw, sh = C.rigid_fit_2d(worlds["side"][ib], worlds["overhead"][ia])
    fixed = side_guess.turned(yaw, sh)
    assert C.angle_deg(fixed.rotation, side.rotation) < 0.1
    assert np.linalg.norm(fixed.position - side.position) < 1.5e-3


# --------------------------------------------------------------------------- yaml write


def test_write_scene_poses_keeps_other_text(tmp_path):
    path = tmp_path / "sim-scene.yaml"
    # Start from a file with no REFINED comment (the real file can already have one).
    text = (REPO / "configs/sim-scene.yaml").read_text().splitlines()
    path.write_text("\n".join(line for line in text if "# REFINED" not in line) + "\n")
    before = path.read_text().splitlines()
    poses = {
        "overhead": C.Pose(C.rot_z(0.1) @ np.diag([1.0, -1.0, -1.0]), np.array([0.6, 0.3, 0.75])),
        "side": look_at(np.array([0.36, -0.47, 0.23]), np.array([0.3, 0.0, 0.0])),
    }
    C.write_scene_poses(path, poses, "test note")
    after = path.read_text().splitlines()
    # Only pos and rotation_cv lines change, and one comment line per entry is new.
    assert len(after) == len(before) + 2
    changed = [line for line in after if line not in before]
    assert sum("# REFINED test note" in line for line in changed) == 2
    assert all(("pos:" in line or "rotation_cv:" in line or "REFINED" in line) for line in changed)
    assert all(line in after for line in before if "pos:" not in line and "rotation_cv" not in line)
    scene = load_scene_config(path)
    for name, pose in poses.items():
        cam = scene.cameras[name]
        assert np.allclose(cam.pos, pose.position, atol=1e-4)
        assert np.allclose(cam.rotation_cv(), pose.rotation, atol=2e-5)
    assert (
        scene.cameras["wrist"].pos
        == load_scene_config(REPO / "configs/sim-scene.yaml").cameras["wrist"].pos
    )
    # A second write replaces the comment line, it does not add another one.
    C.write_scene_poses(path, poses, "second note")
    again = path.read_text().splitlines()
    assert len(again) == len(after)
    assert sum("# REFINED second note" in line for line in again) == 2


# --------------------------------------------------------------------------- real fixture


@pytest.mark.skipif(not (FIXTURE / "meta.json").exists(), reason="fixture-001 is not here")
def test_real_fixture_levelling_and_board_fit():
    settings = C.load_settings()
    scene = load_scene_config(REPO / "configs/sim-scene.yaml")
    frames = {f.camera: f for f in C.load_fixture(FIXTURE)}
    cams = C.level_cameras(frames, scene, settings)
    for name in C.CAMERAS:
        cam = cams[name]
        assert cam["table"].rms_mm < 3.0
        after = C.plane_error(cam["table"], cam["pose1"], (0.3, 0.0))
        assert after["tilt_deg"] < 0.01 and abs(after["height_mm"]) < 0.1
    if not all(p.exists() for p in BOARD_IMAGES.values()):
        pytest.skip("board images are not here")
    board = C.board_fit(cams, BOARD_IMAGES, settings, None)
    assert board["shared_corners"] >= 15
    assert board["fit_residual_mm"]["median"] < 3.0
    for motion in board["camera_motion_vs_fixture"].values():
        assert max(abs(v) for v in motion["median_shift_px"]) < 0.5
