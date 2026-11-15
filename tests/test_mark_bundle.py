"""Bundle checks use fake bytes, never claim trained-model or runtime validation."""

import json
import runpy
from pathlib import Path

import pytest

from brickcell.util import sha256

bundle = runpy.run_path("scripts/bundle_mark_models.py")["bundle"]


def exports(tmp_path):
    for camera, serial in (("plate", "243622060450"), ("wrist", "sonix-wrist")):
        folder = tmp_path / camera
        folder.mkdir()
        (folder / "model.xml").write_text("FAKE IR fixture")
        (folder / "model.bin").write_bytes(b"FAKE weights fixture")
        meta = dict(
            synthetic=False,
            sensor_contract=dict(serial=serial, modality="rgb", crop=[0, 0, 64, 64]),
            model_sha256=sha256(folder / "model.xml"),
            weights_sha256=sha256(folder / "model.bin"),
            checkpoint_sha256="fixture-checkpoint",
            model_version="fixture-version",
            score_type="raw_patchcore_distance_not_probability",
            threshold=dict(
                value=1.0,
                uncertainty_margin=0.02,
                version="fixture-threshold",
                calibration_manifest_sha256="fixture-manifest",
                sample_count=2,
            ),
        )
        files = {
            "model.json": meta,
            "metadata.json": meta,
            "package_versions.json": {"anomalib": "2.7.0"},
            "dataset_validation.json": dict(valid=True, manifest_sha256="fixture-manifest"),
            "parity.json": dict(passed=True, partition="val"),
            "test_report.json": dict(
                partition="held_out_test",
                synthetic=False,
                model_version="fixture-version",
                threshold_version="fixture-threshold",
            ),
        }
        for name, data in files.items():
            (folder / name).write_text(json.dumps(data))
    return tmp_path / "plate", tmp_path / "wrist"


def test_bundle_is_portable_and_hashes_every_handoff_file(tmp_path):
    plate, wrist = exports(tmp_path)
    target = tmp_path / "handoff"
    result = bundle(plate, wrist, target, Path("configs/mark-viewer.json"))
    config = json.loads((target / "configs/mark-viewer.json").read_text())
    for camera in ("plate", "wrist"):
        settings = config["cameras"][camera]
        assert settings["crop"] == [0, 0, 64, 64]
        assert (target / "configs" / settings["models"]["rgb"]).is_file()
    for name, digest in result["files"].items():
        assert sha256(target / name) == digest
    assert result["runtime_compiled_on_packaging_host"] is False


@pytest.mark.parametrize(
    "problem", ["missing_weights", "tampered", "synthetic", "serial", "threshold", "parity", "test"]
)
def test_bundle_rejects_incomplete_or_untrusted_pair_before_writing(tmp_path, problem):
    plate, wrist = exports(tmp_path)
    if problem == "missing_weights":
        (wrist / "model.bin").unlink()
    elif problem == "tampered":
        (wrist / "model.bin").write_bytes(b"changed")
    else:
        name = {"parity": "parity.json", "test": "test_report.json"}.get(problem, "model.json")
        path = wrist / name
        data = json.loads(path.read_text())
        if problem == "synthetic":
            data["synthetic"] = True
        elif problem == "serial":
            data["sensor_contract"]["serial"] = "243622060450"
        elif problem == "threshold":
            data["threshold"]["value"] = float("nan")
        elif problem == "parity":
            data["passed"] = False
        elif problem == "test":
            data["threshold_version"] = "stale"
        path.write_text(json.dumps(data))
    target = tmp_path / "handoff"
    with pytest.raises(ValueError):
        bundle(plate, wrist, target, Path("configs/mark-viewer.json"))
    assert not target.exists()
