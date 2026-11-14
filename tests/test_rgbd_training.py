"""Injected packet fixtures verify preparation; no real accuracy is measured."""

import base64
import io
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from brickcell.data.manifest import load_manifest
from brickcell.data.rgbd import prepare
from brickcell.detector.config import ModelConfig, load_config
from brickcell.detector.workflow import verify_sensor_manifest
from brickcell.inspection.rgbd import DEPTH_ENCODING, encode_depth


def index(tmp_path):
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[8:24, 8:24] = [20, 100, 220]
    depth = np.full((32, 32), 500, np.uint16)
    a, b = io.BytesIO(), io.BytesIO()
    Image.fromarray(rgb).save(a, format="JPEG")
    Image.fromarray(depth).save(b, format="PNG")
    packet = dict(
        synthetic=False,
        aligned_to="color",
        serial="cam",
        frame_id="frame",
        time=100,
        depth_scale_m=0.001,
        rgb_jpeg_b64=base64.b64encode(a.getvalue()).decode(),
        depth_png_b64=base64.b64encode(b.getvalue()).decode(),
    )
    (tmp_path / "snapshot.json").write_text(json.dumps(packet))
    data = dict(
        serial="cam",
        crop=[0, 0, 32, 32],
        captures=[
            dict(
                snapshot="snapshot.json",
                object_id="brick-a",
                session_id="session-a",
                view_id="front",
                label="good",
                split="unsplit",
            )
        ],
    )
    path = tmp_path / "index.json"
    path.write_text(json.dumps(data))
    return path, data, packet


def test_prepared_depth_is_the_exact_inference_encoding_and_groups_match(tmp_path):
    path, _, _ = index(tmp_path)
    result = prepare(path, tmp_path / "prepared")
    a, b = [
        load_manifest(tmp_path / "prepared" / f"{mode}-manifest.json") for mode in ["rgb", "depth"]
    ]
    assert a.samples[0].object_id == b.samples[0].object_id == "brick-a"
    assert a.samples[0].split == b.samples[0].split == "unsplit"
    expected, _ = encode_depth(np.full((32, 32), 500, np.uint16), 0.001)
    np.testing.assert_array_equal(np.asarray(Image.open(b.image_path(b.samples[0]))), expected)
    contract = json.loads(b.path.read_text())["sensor_contract"]
    assert contract["depth_encoding"] == DEPTH_ENCODING
    assert result["trained"] is False
    provenance = json.loads((tmp_path / "prepared/provenance.json").read_text())
    assert provenance["source_captures"][0]["sha256"]


def test_duplicate_images_cannot_be_repackaged_into_different_groups(tmp_path):
    path, data, _ = index(tmp_path)
    data["captures"].append(dict(data["captures"][0], object_id="brick-b", session_id="session-b"))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(path, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_defective_capture_cannot_train(tmp_path):
    path, data, _ = index(tmp_path)
    data["captures"][0].update(label="defective", split="train")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="only good"):
        prepare(path, tmp_path / "prepared")


def test_wrong_camera_and_existing_output_rejected(tmp_path):
    path, data, _ = index(tmp_path)
    data["serial"] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare(path, tmp_path / "prepared")
    with pytest.raises(ValueError, match="new output"):
        prepare(path, tmp_path)


def test_depth_config_contract_verified_against_manifest(tmp_path):
    path, _, _ = index(tmp_path)
    prepare(path, tmp_path / "prepared")
    config = ModelConfig(
        manifest=tmp_path / "prepared/depth-manifest.json",
        output_dir=tmp_path / "model",
        sensor_contract=dict(
            serial="cam", crop=[0, 0, 32, 32], modality="depth", depth_encoding=DEPTH_ENCODING
        ),
    )
    assert verify_sensor_manifest(config)["modality"] == "depth"
    with pytest.raises(ValueError, match="differs"):
        verify_sensor_manifest(replace(config, sensor_contract=None))
    bad = dict(config.sensor_contract, depth_encoding=dict(DEPTH_ENCODING, far_m=2))
    with pytest.raises(ValueError, match="fixed viewer"):
        replace(config, sensor_contract=bad)


def test_shipped_depth_config_is_depth_not_rgb():
    from pathlib import Path

    config = load_config(Path("configs/patchcore-depth.yaml"))
    assert config.sensor_contract["modality"] == "depth"


def test_legacy_rgb_inspector_refuses_depth_model():
    from types import SimpleNamespace

    from brickcell.inspection.service import InspectionService

    detector = SimpleNamespace(metadata={"sensor_contract": {"modality": "depth"}})
    with pytest.raises(ValueError, match="aligned depth"):
        InspectionService(detector)


def test_group_split_preserves_sensor_contract(manifest, tmp_path):
    from dataclasses import asdict, replace

    from brickcell.data.manifest import split_manifest

    rows = [
        asdict(
            replace(s, split="unsplit", session_id=f"session-{i}", path=str(manifest.image_path(s)))
        )
        for i, s in enumerate(manifest.samples)
    ]
    contract = dict(
        serial="test-fixture", crop=[0, 0, 32, 32], modality="depth", depth_encoding=DEPTH_ENCODING
    )
    source = tmp_path / "unsplit.json"
    source.write_text(json.dumps(dict(schema_version=1, samples=rows, sensor_contract=contract)))
    split_manifest(source, tmp_path / "split.json")
    assert json.loads((tmp_path / "split.json").read_text())["sensor_contract"] == contract


def test_object_group_cannot_leak_between_paired_splits(tmp_path):
    path, data, packet = index(tmp_path)
    a, b = io.BytesIO(), io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), 125, np.uint8)).save(a, format="JPEG")
    Image.fromarray(np.full((32, 32), 700, np.uint16)).save(b, format="PNG")
    packet.update(
        frame_id="different",
        rgb_jpeg_b64=base64.b64encode(a.getvalue()).decode(),
        depth_png_b64=base64.b64encode(b.getvalue()).decode(),
    )
    (tmp_path / "second.json").write_text(json.dumps(packet))
    data["captures"][0]["split"] = "train"
    data["captures"].append(
        dict(
            data["captures"][0], snapshot="second.json", session_id="different-session", split="val"
        )
    )
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="split overlap"):
        prepare(path, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
