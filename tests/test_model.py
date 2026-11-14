"""Opt-in, offline after first pretrained download; trained artifacts are prerequisites."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.model


@pytest.fixture
def config():
    from brickcell.detector.config import load_config

    path = os.environ.get("BRICKCELL_MODEL_CONFIG")
    if not path:
        pytest.skip("Set BRICKCELL_MODEL_CONFIG=configs/patchcore.yaml after train/export")
    config = load_config(Path(path))
    if not (config.output_dir / "model.xml").exists():
        pytest.skip("Exported model assets not available")
    return config


def test_openvino_load_and_infer(config):
    from brickcell.data.manifest import load_manifest
    from brickcell.detector.runtime import OpenVINODetector
    from brickcell.inspection.images import read_rgb

    manifest = load_manifest(config.manifest)
    detector = OpenVINODetector(config.output_dir / "model.xml")
    prediction = detector.predict(read_rgb(manifest.image_path(manifest.samples[0])))
    assert prediction.score >= 0
    assert prediction.anomaly_map.shape == config.image_size
    assert detector.threshold is not None


def test_torch_openvino_parity(config):
    from brickcell.detector.workflow import parity

    assert parity(config)["passed"]


def test_real_inspector_drives_mock(config):
    import yaml

    from brickcell.controller.robot import MockRobot
    from brickcell.controller.supervisor import Supervisor
    from brickcell.data.manifest import load_manifest
    from brickcell.detector.runtime import OpenVINODetector
    from brickcell.inspection.images import QualityConfig
    from brickcell.inspection.service import InspectionService
    from brickcell.sources.directory import DirectorySource

    manifest = load_manifest(config.manifest)
    samples = [
        next(s for s in manifest.partition("test") if s.label == label)
        for label in ("good", "defective")
    ]
    source = DirectorySource({s.image_id: [manifest.image_path(s)] for s in samples})
    quality = QualityConfig(
        **yaml.safe_load(
            Path(os.environ.get("BRICKCELL_QUALITY_CONFIG", "configs/inspection.yaml")).read_text()
        )
    )
    report = Supervisor(
        MockRobot(),
        source,
        InspectionService(OpenVINODetector(config.output_dir / "model.xml"), quality),
    ).run(list(source.frames))
    assert report["completed_mock_cycles"] == 2
    assert report["inspector_mode"] == "anomalib_openvino_cpu"


def test_same_object_id_changes_bowl_when_camera_pixels_change(config):
    """Causality regression: neither object ID nor filename selects the destination."""
    import yaml

    from brickcell.controller.robot import MockRobot
    from brickcell.controller.supervisor import Supervisor
    from brickcell.data.manifest import load_manifest
    from brickcell.detector.runtime import OpenVINODetector
    from brickcell.inspection.images import QualityConfig
    from brickcell.inspection.service import InspectionService
    from brickcell.sources.directory import DirectorySource

    manifest = load_manifest(config.manifest)
    quality = QualityConfig(
        **yaml.safe_load(
            Path(os.environ.get("BRICKCELL_QUALITY_CONFIG", "configs/inspection.yaml")).read_text()
        )
    )
    inspector = InspectionService(OpenVINODetector(config.output_dir / "model.xml"), quality)
    for label, expected in [("good", "move_good"), ("defective", "move_defective")]:
        sample = next(s for s in manifest.partition("test") if s.label == label)
        source = DirectorySource({"same-opaque-object": [manifest.image_path(sample)]})
        robot = MockRobot()
        report = Supervisor(robot, source, inspector).run(["same-opaque-object"])
        assert report["completed_cycles"] == 1
        moves = [a.operation for a in robot.actions if a.operation.startswith("move_")]
        assert moves == [expected]
