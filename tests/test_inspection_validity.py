"""Offline RGB-D evidence tests; no hardware, HTTP clients or optional simulator imports."""

import ast
import concurrent.futures
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from brickcell.inspection.validity import validate_plate_view

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/inspection_validity/hardware_cell.py.txt"
PATCH = ROOT / "patches/inspection-validity/hardware_cell.patch"


def pixels(marked=False):
    rgb = np.full((80, 80, 3), [240, 190, 40], dtype=np.uint8)
    if marked:
        rgb[38:41, 20:60] = 25
    return rgb


def evidence(rgb=None, depth=None, **kwargs):
    return validate_plate_view(
        pixels() if rgb is None else rgb,
        np.full((80, 80), 0.28, dtype=np.float32) if depth is None else depth,
        kwargs.pop("crop_box", [0, 0, 80, 80]),
        kwargs.pop("expected_z_m", 0.28),
        kwargs.pop("z_band_m", 0.075),
        **kwargs,
    )


def test_depth_support_is_explicit_and_not_crop_area_fallback():
    good = evidence()
    assert good.valid_view and good.depth_supported
    assert good.foreground_pixels == 6400
    missing = validate_plate_view(pixels(), None, [0, 0, 80, 80], 0.28, 0.075)
    assert not missing.valid_view and not missing.depth_supported
    assert missing.reason == "missing_depth"
    assert missing.foreground_pixels == 0


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, 0.6, 280.0])
def test_invalid_or_background_depth(value):
    result = evidence(depth=np.full((80, 80), value, dtype=np.float32))
    assert not result.valid_view and not result.depth_supported
    assert result.reason == "insufficient_depth_support"


@pytest.mark.parametrize(
    "depth", [np.zeros((40, 80)), np.zeros((80, 80, 1)), np.full((80, 80), 280, dtype=np.uint16)]
)
def test_depth_shape_and_metric_float_contract(depth):
    assert evidence(depth=depth).reason == "invalid_aligned_metric_depth"


@pytest.mark.parametrize(
    "box",
    [
        [-1, 0, 80, 80],
        [0, 0, 81, 80],
        [80, 0, 0, 80],
        [0, 0, 0, 80],
        [0.0, 0, 80, 80],
        [True, 0, 80, 80],
        [0, 0, 80],
        None,
    ],
)
def test_crop_validation(box):
    result = evidence(crop_box=box)
    assert not result.valid_view and "crop" in result.reason


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((80, 80)),
        np.zeros((80, 80, 4), dtype=np.uint8),
        np.zeros((80, 80, 3)),
        np.zeros((0, 80, 3), dtype=np.uint8),
    ],
)
def test_malformed_images(rgb):
    assert not evidence(rgb=rgb).valid_view


@pytest.mark.parametrize("expected", [None, 0, -1, np.nan, np.inf, "bad"])
def test_expected_depth_required(expected):
    assert evidence(expected_z_m=expected).reason == "invalid_expected_depth"


@pytest.mark.parametrize("band", [0, -1, np.nan, np.inf, 0.28, 0.5])
def test_invalid_depth_band(band):
    assert not evidence(z_band_m=band).valid_view


def test_sparse_or_fragmented_depth_is_not_enough():
    depth = np.zeros((80, 80), dtype=np.float32)
    depth[30:38, 30:38] = 0.28  # 64 pixels cleared the previous 45px-only check.
    assert evidence(depth=depth).reason == "insufficient_depth_support"
    depth[::2, ::2] = 0.28
    assert evidence(depth=depth).reason == "no_inspectable_surface"


@pytest.mark.parametrize("hue", [100, 130])
def test_plate_or_jaw_only_cannot_be_classified_good(hue):
    hsv = np.full((80, 80, 3), [hue, 220, 200], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    assert evidence(rgb=rgb).reason == "no_inspectable_surface"


def test_uniform_empty_image_and_tiny_crop_rejected():
    assert not evidence(rgb=np.full((80, 80, 3), 100, dtype=np.uint8)).valid_view
    assert not evidence(crop_box=[0, 0, 19, 20]).valid_view


@pytest.fixture(scope="module")
def patched_source(tmp_path_factory):
    folder = tmp_path_factory.mktemp("deployed-inspection-patch")
    target = folder / "src/brickcell/block_inspector/hardware_cell.py"
    target.parent.mkdir(parents=True)
    target.write_text(FIXTURE.read_text())
    subprocess.run(
        ["git", "apply", "--check", str(PATCH)], cwd=folder, check=True, capture_output=True
    )
    subprocess.run(["git", "apply", str(PATCH)], cwd=folder, check=True, capture_output=True)
    return target.read_text()


@dataclass
class Outcome:
    succeeded: bool
    reason: str
    metadata: dict | None = None


class StaleFrame(Exception):
    pass


def load_definitions(source):
    # Execute exact deployed definitions, not imports/constructors that pull in
    # MuJoCo/Frax or live arm clients. The sole executor method uses injected fakes.
    tree = ast.parse(source)
    names = {
        "score_brick_roi_dark_mark",
        "FrameInspectionResult",
        "RealCameraAnomalyInspector",
        "PhysicalSO101CellExecutor",
    }
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    namespace = dict(
        np=np,
        cv2=cv2,
        math=math,
        concurrent=concurrent,
        Any=Any,
        dataclass=dataclass,
        validate_plate_view=validate_plate_view,
        SkillOutcome=Outcome,
        StaleCameraFrameError=StaleFrame,
        time=SimpleNamespace(sleep=lambda _: None),
        PHYSICAL_PLATE_SCAN_POS=np.array([0.312, -0.025, 0.118]),
        PHYSICAL_JOINT5_CCW_START_RAD=0.45,
        PHYSICAL_JOINT5_CCW_END_RAD=3.05,
        PHYSICAL_OBSERVATION_HOME_Q=np.array([0, -1.15, 0.88, 0.42, 0.55]),
        PART_ID_TO_DASHBOARD_ID={"3437": "part-001"},
        project_world_obb_to_image_box=lambda *a, **kw: ([0, 0, 80, 80], 0.28),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(FIXTURE), "exec"), namespace)
    return namespace


def run_scan(
    source,
    *,
    depth=True,
    marked=False,
    repeat=False,
    stale=False,
    valid_count=7,
    dry_run=False,
    gripper=0.0,
):
    ns = load_definitions(source)
    arm_calls = []
    inspector = ns["RealCameraAnomalyInspector"]()
    obj = ns["PhysicalSO101CellExecutor"]()
    obj.arm = SimpleNamespace(
        dry_run=dry_run,
        current_q_and_grip=lambda **kw: (np.array([0, -1, 0.9, 0.4, 0.5]), gripper),
        move_to_pose=lambda *a, **kw: arm_calls.append((a, kw)),
    )
    obj.ik = SimpleNamespace(
        solve_grasp_pose=lambda *a, **kw: SimpleNamespace(
            q_grasp=np.array([0.0, -1.0, 0.9, 0.4, 0.5])
        ),
        cal_kin=SimpleNamespace(
            calibration=SimpleNamespace(lower=np.full(5, -3.1), upper=np.full(5, 3.1))
        ),
        fk_world_body3=lambda q: np.eye(4),
    )
    count = 0

    def snapshot(camera="plate"):
        nonlocal count
        if stale:
            raise StaleFrame("expired test frame")
        if camera == "plate":
            count += 1
        dep = (
            np.full((80, 80), 0.28 if count <= valid_count else 0.6, dtype=np.float32)
            if depth
            else None
        )
        return SimpleNamespace(
            rgb=pixels(marked), depth_m=dep, frame_id="repeat" if repeat else str(count)
        )

    obj.frame_source = SimpleNamespace(
        fetch_d415_snapshot=snapshot, fetch_wrist_snapshot=lambda: snapshot("wrist")
    )
    obj.inspector = inspector
    obj._active_brick = SimpleNamespace(
        spec=SimpleNamespace(part_id="3437", name="test brick", dims_m=[0.03] * 3),
        T_world_block=np.eye(4),
    )
    obj.hand_empty = False
    obj._active_close_pct = 0.2
    obj._T_body3_block_held = np.eye(4)
    obj.dashboard = None
    obj.real_cams = {"plate": (np.eye(3), np.zeros(3), np.eye(3))}
    obj.plate_scan_keyframes = []
    obj.plate_scan_completed = False
    obj.anomaly_untrusted_error = None
    obj.max_plate_anomaly_distance = None
    obj.anomaly_flagged = False
    obj.anomaly_messages = []
    obj._sync_mirror_and_telemetry = lambda *a: None
    obj._project_to_top_box = lambda *a: [0, 0, 80, 80]
    try:
        result = obj._skill_present_and_spin_at_plate_camera()
        return result, obj, arm_calls
    finally:
        inspector.close()


def test_reproduces_original_bug_and_blocks_it_after_patch(patched_source):
    original, _, _ = run_scan(FIXTURE.read_text(), depth=False)
    assert original.succeeded
    fixed, obj, _ = run_scan(patched_source, depth=False, gripper=0.9)
    assert not fixed.succeeded
    assert fixed.reason == "missing_depth_supported_inspection_view"
    assert obj.plate_scan_valid_frames == 0 and not obj.plate_scan_completed


@pytest.mark.parametrize("dry_run", [False, True])
def test_gripper_or_dry_run_never_substitutes_for_evidence(patched_source, dry_run):
    result, _, _ = run_scan(patched_source, depth=False, gripper=0.9, dry_run=dry_run)
    assert not result.succeeded


@pytest.mark.parametrize("marked", [False, True])
def test_real_inspector_and_complete_fake_presentation_loop(patched_source, marked):
    result, obj, calls = run_scan(patched_source, marked=marked)
    assert result.succeeded and obj.plate_scan_completed
    assert result.metadata["valid_plate_views"] == 7
    assert result.metadata["depth_supported_view_seen"]
    assert obj.anomaly_flagged is marked
    assert len(calls) == 9  # existing carry + presentation + seven rotation poses


def test_background_depth_blocks_scan_completion(patched_source):
    result, obj, _ = run_scan(patched_source, valid_count=0)
    assert not result.succeeded and obj.plate_scan_valid_frames == 0


def test_two_valid_views_do_not_complete_scan(patched_source):
    result, obj, _ = run_scan(patched_source, valid_count=2)
    assert not result.succeeded and obj.plate_scan_valid_frames == 2
    assert result.reason == "insufficient_valid_plate_views(2)"


def test_repeated_and_stale_frames_do_not_complete_scan(patched_source):
    repeated, obj, _ = run_scan(patched_source, repeat=True)
    assert not repeated.succeeded and obj.plate_scan_valid_frames == 1
    stale, _, _ = run_scan(patched_source, stale=True)
    assert not stale.succeeded and stale.reason.startswith("stale_camera_frame:")


def test_invalid_view_never_calls_anomaly_model(patched_source):
    class ForbiddenModel:
        threshold = SimpleNamespace(value=0.5)

        def predict(self, image):
            pytest.fail("Invalid view reached the model")

    ns = load_definitions(patched_source)
    inspector = ns["RealCameraAnomalyInspector"](openvino_detector=ForbiddenModel())
    try:
        result = inspector.evaluate_frame(pixels(), None, [0, 0, 80, 80], expected_z_m=0.28)
        assert not result.valid_view and not result.depth_supported
    finally:
        inspector.close()


def test_patch_preserves_all_motion_calls_and_dark_mark_algorithm(patched_source):
    def selected(source):
        tree = ast.parse(source)
        scorer = next(
            n for n in tree.body if getattr(n, "name", None) == "score_brick_roi_dark_mark"
        )
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "move_to_pose"
        ]
        return ast.dump(scorer), [ast.dump(n) for n in calls]

    assert selected(FIXTURE.read_text()) == selected(patched_source)


def test_patch_preparation_refuses_unknown_source_and_overwrites(tmp_path):
    source = tmp_path / "hardware_cell.py"
    source.write_text("# teammate has changed this controller\n")
    out = tmp_path / "review.py"
    command = [
        "python3",
        str(ROOT / "scripts/prepare_inspection_validity_patch.py"),
        "--source",
        str(source),
        "--output",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and "differs from the reviewed" in result.stderr
    assert not out.exists()
    out.write_text("preserve me")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and "must be a new file" in result.stderr
    assert out.read_text() == "preserve me"
