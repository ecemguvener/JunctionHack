import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from brickcell.inspection.mark_scan import MarkScanEvidence
from brickcell.inspection.types import Decision


def record(scan, camera, stage, valid=True, flagged=False):
    scan.record(camera, stage, f"{camera}-{stage}", valid=valid, flagged=flagged)


def test_plate_alone_cannot_complete_mark_scan():
    scan = MarkScanEvidence()
    for stage in range(7):
        record(scan, "plate", stage)
    assert scan.decision() == Decision.REINSPECT
    assert scan.missing == [("wrist", 0), ("wrist", 3), ("wrist", 6)]
    for stage in (0, 3, 6):
        record(scan, "wrist", stage)
    assert scan.decision() == Decision.GOOD


def test_one_uncertain_view_prevents_clean_completion():
    scan = MarkScanEvidence()
    for camera, stage in sorted(scan.required):
        record(scan, camera, stage, valid=(camera, stage) != ("wrist", 3))
    assert scan.decision() == Decision.REINSPECT
    assert scan.missing == [("wrist", 3)]


@pytest.mark.parametrize("camera", ["plate", "wrist"])
def test_trusted_mark_in_either_camera_is_preserved(camera):
    scan = MarkScanEvidence()
    record(scan, camera, 0, flagged=True)
    record(scan, camera, 3)
    assert scan.decision() == Decision.DEFECTIVE


def test_invalid_flag_and_previous_brick_do_not_supply_defect_evidence():
    scan = MarkScanEvidence()
    record(scan, "plate", 0, valid=False, flagged=True)
    assert scan.decision() == Decision.REINSPECT
    record(scan, "wrist", 0, flagged=True)
    assert scan.decision() == Decision.DEFECTIVE
    assert MarkScanEvidence().decision() == Decision.REINSPECT


def test_duplicate_frames_or_stages_cannot_create_view_coverage():
    scan = MarkScanEvidence()
    record(scan, "plate", 0)
    with pytest.raises(ValueError, match="repeated"):
        scan.record("plate", 1, "plate-0", valid=True, flagged=False)
    with pytest.raises(ValueError, match="already recorded"):
        scan.record("plate", 0, "new", valid=True, flagged=False)
    with pytest.raises(ValueError, match="Unexpected"):
        scan.record("top", 0, "new", valid=True, flagged=False)


def test_actual_scan_preflight_requires_both_models_before_any_motion():
    """Execute the actual method without importing optional robot/simulator SDKs."""
    source = ast.parse(Path("src/brickcell/block_inspector/hardware_cell.py").read_text())
    executor = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "PhysicalSO101CellExecutor"
    )
    method = next(
        node
        for node in executor.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_skill_present_and_spin_at_plate_camera"
    )
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + [method], type_ignores=[]
    )
    namespace = {"SkillOutcome": lambda success, **kw: SimpleNamespace(succeeded=success, **kw)}
    exec(compile(module, "<actual-scan-method>", "exec"), namespace)
    # Deliberately has no arm/IK: access would fail before the missing-model guard.
    cell = SimpleNamespace(
        hand_empty=False,
        _active_brick=object(),
        plate_scan_completed=True,
        inspector=SimpleNamespace(has_detector_for_camera=lambda camera: camera == "plate"),
    )
    result = namespace[method.name](cell)
    assert not result.succeeded
    assert result.reason == "missing_mark_models(wrist)"
    assert not cell.plate_scan_completed
