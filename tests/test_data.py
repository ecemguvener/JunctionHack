import json
from dataclasses import asdict, replace

import pytest

from brickcell.data.manifest import (
    Manifest,
    import_directory,
    load_manifest,
    split_manifest,
    validate,
)
from brickcell.util import save_json


def test_synthetic_dataset_valid(manifest):
    report = validate(manifest)
    assert report["valid"]
    assert report["counts"]["train/good"] == 12
    assert all(s.label == "good" for s in manifest.partition("train"))
    assert len(manifest.partition("test")) == 12


def test_duplicate_hash_and_paths(manifest):
    sample = manifest.samples[0]
    manifest.samples.append(replace(sample, image_id="duplicate", split="test"))
    errors = validate(manifest)["errors"]
    assert any("Duplicate file hash" in e for e in errors)
    assert any("Repeated image path" in e for e in errors)
    assert any("Split overlap" in e for e in errors)


def test_corrupt_and_missing(manifest):
    manifest.image_path(manifest.samples[0]).write_bytes(b"not an image")
    manifest.image_path(manifest.samples[1]).unlink()
    assert sum("missing/corrupt" in e for e in validate(manifest)["errors"]) == 2


def test_duplicate_id_and_anomalous_training(manifest):
    manifest.samples[0] = replace(
        manifest.samples[0], label="defective", image_id=manifest.samples[1].image_id
    )
    errors = validate(manifest)["errors"]
    assert any("training must" in e for e in errors)
    assert any("Duplicate or empty" in e for e in errors)


def test_object_and_session_leakage(manifest):
    manifest.samples[-1] = replace(
        manifest.samples[-1],
        object_id=manifest.samples[0].object_id,
        session_id=manifest.samples[0].session_id,
    )
    errors = validate(manifest)["errors"]
    assert any("object_id" in e for e in errors)
    assert any("session_id" in e for e in errors)


def test_group_split_deterministic(manifest, tmp_path):
    rows = [
        asdict(
            replace(s, split="unsplit", session_id=f"session-{i}", path=str(manifest.image_path(s)))
        )
        for i, s in enumerate(manifest.samples)
    ]
    source = tmp_path / "unsplit.json"
    save_json(source, {"schema_version": 1, "samples": rows})
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert split_manifest(source, first)["valid"]
    assert split_manifest(source, second)["valid"]
    assert json.loads(first.read_text()) == json.loads(second.read_text())
    assert all(s.label == "good" for s in load_manifest(first).partition("train"))
    with pytest.raises(ValueError, match="reshuffle"):
        split_manifest(first, second)


def test_group_split_rejects_insufficient_groups(manifest, tmp_path):
    rows = [
        asdict(
            replace(s, split="unsplit", session_id="one-session", path=str(manifest.image_path(s)))
        )
        for s in manifest.samples
    ]
    source = tmp_path / "unsplit.json"
    save_json(source, {"schema_version": 1, "samples": rows})
    with pytest.raises(ValueError, match="independent"):
        split_manifest(source, tmp_path / "out.json")


def test_directory_import_requires_metadata(tmp_path):
    from PIL import Image

    image = tmp_path / "raw" / "good" / "capture1" / "brick1" / "front" / "image.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(image)
    output = tmp_path / "manifest.json"
    import_directory(tmp_path / "raw", output)
    sample = load_manifest(output).samples[0]
    assert (sample.object_id, sample.session_id, sample.view_id) == ("brick1", "capture1", "front")
    assert sample.split == "unsplit"
    Image.new("RGB", (32, 32)).save(tmp_path / "raw" / "wrong.png")
    with pytest.raises(ValueError, match="Expected label"):
        import_directory(tmp_path / "raw", output)


def test_bad_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 4}')
    with pytest.raises(ValueError, match="schema"):
        load_manifest(path)


def test_missing_class_and_empty(manifest):
    assert not validate(Manifest(manifest.path, []))["valid"]
    manifest.samples = [s for s in manifest.samples if s.label != "defective"]
    assert not validate(manifest)["valid"]


def test_changed_image_bytes_rejected_after_training(manifest, tmp_path):
    from brickcell.detector.config import ModelConfig
    from brickcell.detector.workflow import verify_dataset_snapshot
    from brickcell.util import sha256

    config = ModelConfig(manifest.path, tmp_path / "model")
    save_json(
        config.output_dir / "data_provenance.json",
        {
            "images": [
                {"image_id": s.image_id, "split": s.split, "sha256": sha256(manifest.image_path(s))}
                for s in manifest.samples
            ]
        },
    )
    verify_dataset_snapshot(config, manifest)
    path = manifest.image_path(manifest.samples[-1])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed since training"):
        verify_dataset_snapshot(config, manifest)
