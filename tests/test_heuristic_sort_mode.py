"""Exercise opt-in physical routing policy without importing or commanding hardware."""

import ast
import concurrent.futures
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from brickcell.inspection.dark_marks import score_brick_roi_dark_mark
from brickcell.inspection.mark_scan import MarkScanEvidence
from brickcell.inspection.types import Decision


def inspector_type():
    source = ast.parse(Path("src/brickcell/block_inspector/hardware_cell.py").read_text())
    classes = [
        n
        for n in source.body
        if isinstance(n, ast.ClassDef)
        and n.name in ("FrameInspectionResult", "RealCameraAnomalyInspector")
    ]
    namespace = dict(
        dataclass=dataclass,
        np=np,
        math=math,
        concurrent=concurrent,
        score_brick_roi_dark_mark=score_brick_roi_dark_mark,
    )
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + classes, type_ignores=[]
    )
    exec(compile(module, "<actual-heuristic-inspector>", "exec"), namespace)
    return namespace["RealCameraAnomalyInspector"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("marked", [False, True])
def test_lab_rule_only_authorizes_results_when_explicitly_selected(enabled, marked):
    service = inspector_type()(allow_heuristic_sorting=enabled)
    image = np.full((96, 96, 3), [212, 174, 58], np.uint8)
    if marked:
        cv2.line(image, (30, 55), (65, 57), (24, 22, 20), 2)
    try:
        result = service.evaluate_frame(image, None, [0, 0, 96, 96])
        assert result.valid_view is enabled
        assert result.flagged is (enabled and marked)
        assert service.has_detector_for_camera("wrist") is False
        assert service.has_detector_for_camera("plate") is enabled
        if enabled:
            assert service.inspector_mode == "lab_dark_mark_plate_sort"
    finally:
        service.close()


@pytest.mark.parametrize("threshold", [None, 0, -1, float("nan"), float("inf")])
def test_missing_or_invalid_lab_threshold_fails_before_inspection(threshold):
    with pytest.raises(ValueError, match="positive finite"):
        inspector_type()(dark_mark_threshold=threshold, allow_heuristic_sorting=True)


@pytest.mark.parametrize("problem", ["wrist", "empty", "clipped"])
def test_untrusted_views_never_supply_a_sort_result(problem):
    service = inspector_type()(allow_heuristic_sorting=True)
    image = np.full((96, 96, 3), [212, 174, 58], np.uint8)
    if problem == "empty":
        image[:] = 0
    try:
        result = service.evaluate_frame(
            image,
            None,
            [0, 0, 200, 200] if problem == "clipped" else [0, 0, 96, 96],
            camera_name="wrist" if problem == "wrist" else "plate",
        )
        assert not result.valid_view and not result.flagged
    finally:
        service.close()


def test_explicit_plate_policy_requires_all_scheduled_clean_views():
    scan = MarkScanEvidence(required=frozenset(("plate", i) for i in range(7)))
    for i in range(6):
        scan.record("plate", i, f"frame-{i}", valid=True, flagged=False)
    assert scan.decision() == Decision.REINSPECT
    scan.record("plate", 6, "frame-6", valid=True, flagged=False)
    assert scan.decision() == Decision.GOOD


def test_cannot_silently_mix_lab_rule_with_loaded_models():
    with pytest.raises(ValueError, match="mix heuristic"):
        inspector_type()(
            allow_heuristic_sorting=True, camera_detectors={"plate": {"rgb": object()}}
        )


def deployment_function(inspector):
    source = ast.parse(Path("src/brickcell/block_inspector/hardware_cell.py").read_text())
    method = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_physical_cell_deployment"
    )

    def forbidden(**kwargs):
        pytest.fail("Explicit Lab mode attempted model discovery or hardware access")

    namespace = dict(
        Path=Path,
        RealCameraAnomalyInspector=inspector,
        load_anomalib_camera_detectors=forbidden,
        LiveBenchHttpFrameSource=forbidden,
    )
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + [method], type_ignores=[]
    )
    exec(compile(module, "<actual-lab-deployment-entry>", "exec"), namespace)
    return namespace[method.name]


def test_default_deployment_selects_lab_rule_without_discovering_models_or_opening_hardware(
    tmp_path,
):
    class ReachedInspector(Exception):
        pass

    def inspector(**kwargs):
        assert kwargs["allow_heuristic_sorting"] is True
        assert kwargs["camera_detectors"] == {}
        assert kwargs["camera_crops"] == {}
        assert kwargs["dark_mark_threshold"] == 16
        raise ReachedInspector

    with pytest.raises(ReachedInspector):
        deployment_function(inspector)(output_dir=tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(inspection_mode="invalid"),
        dict(inspection_mode="lab-dark-mark", require_anomalib=True),
        dict(inspection_mode="lab-dark-mark", openvino_config_path="model.xml"),
    ],
)
def test_unknown_or_conflicting_modes_fail_before_constructors(tmp_path, kwargs):
    def inspector(**kwargs):
        pytest.fail("Invalid selection reached inspector constructor")

    with pytest.raises(ValueError):
        deployment_function(inspector)(output_dir=tmp_path, **kwargs)


def test_patchcore_plate_mode_allows_calibrated_plate_only_and_rejects_missing_plate(tmp_path):
    class ReachedHardware(Exception):
        pass

    created = []
    source = ast.parse(Path("src/brickcell/block_inspector/hardware_cell.py").read_text())
    method = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_physical_cell_deployment"
    )

    def build_deploy(detectors):
        def inspector(**kwargs):
            obj = SimpleNamespace(**kwargs)
            created.append(obj)
            return obj

        def stop_at_hardware(**kwargs):
            raise ReachedHardware

        namespace = dict(
            Path=Path,
            RealCameraAnomalyInspector=inspector,
            load_anomalib_camera_detectors=lambda **kwargs: (
                detectors,
                {"plate": [0, 0, 640, 480]},
            ),
            LiveBenchHttpFrameSource=stop_at_hardware,
        )
        module = ast.Module(
            body=ast.parse("from __future__ import annotations").body + [method], type_ignores=[]
        )
        exec(compile(module, "<actual-patchcore-plate-entry>", "exec"), namespace)
        return namespace[method.name]

    with pytest.raises(ReachedHardware):
        build_deploy({"plate": {"rgb": object()}})(
            output_dir=tmp_path, inspection_mode="patchcore-plate"
        )
    assert len(created) == 1
    assert created[0].allow_heuristic_sorting is False
    assert created[0].plate_only_anomalib is True

    with pytest.raises(ValueError, match=r"missing_mark_models\(plate\)"):
        build_deploy({})(output_dir=tmp_path, inspection_mode="patchcore-plate")


def test_cli_deploy_cell_defaults_to_patchcore_plate():
    from brickcell.block_inspector.cli import build_parser

    args = build_parser().parse_args(["deploy-cell"])
    assert args.inspection_mode == "patchcore-plate"
