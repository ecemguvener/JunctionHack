"""Train, calibrate, export to OpenVINO CPU, check parity, and evaluate plate PatchCore on Intel."""

import json
import platform
from dataclasses import asdict
from pathlib import Path

from brickcell.data.manifest import Manifest, Sample, require_valid
from brickcell.detector.config import ModelConfig
from brickcell.detector.runtime import OpenVINODetector
from brickcell.detector.workflow import evaluate, export, parity, train
from brickcell.util import save_json, sha256, versions

run = Path(__file__).resolve().parent
artifacts_dir = run.parent
combined_run = artifacts_dir / "patchcore-combined-earlier650-v1"
inventory = json.loads((combined_run / "inventory.json").read_text())
assert len(inventory["images"]) == 650

hashes: set[str] = set()
for row in inventory["images"]:
    assert sha256(combined_run / row["path"]) == row["sha256"]
    assert row["sha256"] not in hashes
    hashes.add(row["sha256"])

jack = artifacts_dir / "imported-datasets/jack-anomaly-data-positive/positive"
jack_rows = [json.loads(line) for line in (jack / "index.jsonl").read_text().splitlines()]
assert len(jack_rows) == 570
for row in jack_rows:
    assert row["condition"] == row["label"] == "positive"
    assert row["synthetic"] is False
    assert row["camera"] == "plate" and row["camera_serial"] == "243622060450"
    digest = sha256(jack / row["path"])
    assert digest not in hashes, "Duplicate within or across earlier and Jack collections"
    hashes.add(digest)
    row["sha256"] = digest

camera, serial = "plate", "243622060450"
sensor_contract = {"serial": serial, "crop": [0, 0, 640, 480], "modality": "rgb"}
output = run / camera
output.mkdir(exist_ok=False)

samples: list[Sample] = []
provenance: list[dict] = []

# 1. Earlier 650 plate images:
# - live-marks-20261114T235451Z accepted clean references (90 good) -> train
# - brick-002-controlled-depth-20261114-a clean (12 good) + marked (12 defective) -> val
for row in inventory["images"]:
    m = row["metadata"]
    if m.get("camera") != camera:
        continue
    if row["disposition"] == "accepted_clean_reference_only":
        assert m["human_label"] == "good" and m["baseline_accepted"] is True
        assert m["serial"] == serial and m["synthetic"] is False
        split = "val" if m["object_id"] == "brick-002" else "train"
        samples.append(
            Sample(
                image_id=row["image_id"],
                path=str((combined_run / row["path"]).resolve()),
                label="good",
                split=split,
                object_id=m["object_id"],
                session_id=m["session_id"],
                view_id=m.get("view_id", camera),
                synthetic=False,
            )
        )
        provenance.append({**row, "assigned_split": split})
    elif (
        row["disposition"] == "excluded_marked_brick"
        and m.get("object_id") == "brick-002"
        and m.get("human_label") == "defective"
    ):
        assert m["serial"] == serial and m["synthetic"] is False
        samples.append(
            Sample(
                image_id=row["image_id"],
                path=str((combined_run / row["path"]).resolve()),
                label="defective",
                split="val",
                object_id=m["object_id"],
                session_id=m["session_id"],
                view_id=m.get("view_id", camera),
                synthetic=False,
            )
        )
        provenance.append({**row, "assigned_split": "val"})

# 2. Jack 570 clean plate images -> train
for i, row in enumerate(jack_rows):
    path = (jack / row["path"]).resolve()
    samples.append(
        Sample(
            image_id=f"jack-{i:04d}",
            path=str(path),
            label="good",
            split="train",
            object_id=row["object_id"],
            session_id=row["session_id"],
            view_id=row["grip_id"],
            family=row["brick_type"],
            color=row["color"],
            synthetic=False,
        )
    )
    provenance.append({**row, "image_id": samples[-1].image_id, "assigned_split": "train"})

# 3. ~/brickcell-captures/raw/{good,defective}/*/side/*.png
raw_dir = Path.home() / "brickcell-captures/raw"
reports_dir = Path.home() / "brickcell-captures/reports"
for png_path in sorted(raw_dir.rglob("*.png")):
    rel = png_path.relative_to(raw_dir)
    if len(rel.parts) != 5 or rel.parts[3] != "side":
        continue
    folder_label, session_id, object_id, cam_name, _ = rel.parts
    report_path = reports_dir / f"{png_path.stem}.json"
    report = json.loads(report_path.read_text())
    assert report.get("human_label") == folder_label
    assert report.get("synthetic") is False
    digest = sha256(png_path)
    assert digest not in hashes, f"Duplicate hash for {png_path}"
    hashes.add(digest)
    if object_id == "blue-brick-001":
        split = "train"
    elif object_id == "brick-002":
        split = "val"
    else:
        split = "test"
    samples.append(
        Sample(
            image_id=f"raw-{png_path.stem[:12]}",
            path=str(png_path.resolve()),
            label=folder_label,
            split=split,
            object_id=object_id,
            session_id=session_id,
            view_id="side",
            synthetic=False,
        )
    )
    provenance.append(
        {
            "image_id": samples[-1].image_id,
            "path": str(png_path.resolve()),
            "sha256": digest,
            "label": folder_label,
            "assigned_split": split,
            "object_id": object_id,
            "session_id": session_id,
            "report_path": str(report_path.resolve()),
        }
    )

# 4. Matched clean & raised defective brick-001 640x480 plate captures -> test
for sub, label in [
    ("brick-001-matched-clean-GhBs4I", "good"),
    ("brick-001-raised-DGMCha", "defective"),
]:
    img_path = (Path.home() / "brickcell-captures" / sub / "243622060450-color.png").resolve()
    digest = sha256(img_path)
    assert digest not in hashes, f"Duplicate hash for {img_path}"
    hashes.add(digest)
    samples.append(
        Sample(
            image_id=sub,
            path=str(img_path),
            label=label,
            split="test",
            object_id="brick-001",
            session_id=sub,
            view_id="plate",
            synthetic=False,
        )
    )
    provenance.append(
        {
            "image_id": sub,
            "path": str(img_path),
            "sha256": digest,
            "label": label,
            "assigned_split": "test",
            "object_id": "brick-001",
            "session_id": sub,
        }
    )

assert len(samples) == 707
manifest_path = output / "manifest.json"
save_json(
    manifest_path,
    {
        "schema_version": 1,
        "sensor_contract": sensor_contract,
        "surface_profile": None,
        "samples": [asdict(s) for s in samples],
    },
)
manifest = Manifest(manifest_path, samples)
validation_report = require_valid(manifest)
save_json(output / "dataset-audit.json", validation_report)
save_json(output / "provenance.json", provenance)
save_json(run / "versions.json", versions())
save_json(
    run / "input-audit.json",
    {
        "total_plate_samples": len(samples),
        "partition_counts": validation_report["counts"],
        "unique_hashes_checked": len(hashes),
        "top_refs_ignored": True,
        "manifest_sha256": validation_report["manifest_sha256"],
        "script_sha256": sha256(Path(__file__)),
    },
)

config = ModelConfig(
    manifest=manifest_path,
    output_dir=output,
    sensor_contract=sensor_contract,
)
metadata = train(config)
export_info = export(config)
parity_report = parity(config)
assert parity_report["passed"] is True, f"Parity failed: {parity_report}"
test_report = evaluate(config)

detector = OpenVINODetector(output / "model.xml")
assert detector.threshold is not None
assert detector.metadata.get("synthetic") is False

summary = {
    "status": "TRAINED_CALIBRATED_EXPORTED_EVALUATED",
    "camera": camera,
    "serial": serial,
    "scope": "CALIBRATED_PLATE_OPENVINO_EVALUATED",
    "device": "cpu",
    "host": platform.node(),
    "architecture": "Anomalib 2.7.0 PatchCore",
    "top_refs_ignored": True,
    "partition_counts": validation_report["counts"],
    "reference_count": validation_report["counts"]["train/good"],
    "validation_count": (
        validation_report["counts"]["val/good"] + validation_report["counts"]["val/defective"]
    ),
    "test_count": (
        validation_report["counts"]["test/good"] + validation_report["counts"]["test/defective"]
    ),
    "checkpoint": str(output / "model.ckpt"),
    "checkpoint_sha256": metadata["checkpoint_sha256"],
    "openvino_xml": export_info["model"],
    "openvino_xml_sha256": sha256(Path(export_info["model"])),
    "openvino_bin_sha256": sha256(Path(export_info["weights"])),
    "manifest_sha256": validation_report["manifest_sha256"],
    "training_seconds": metadata["training_seconds"],
    "threshold": metadata["threshold"],
    "automatic_sorting_allowed": True,
    "accuracy_evaluated": True,
    "parity_passed": parity_report["passed"],
    "parity_max_abs_diff": parity_report["max_absolute_difference"],
    "validation_metrics": metadata["validation_metrics"],
    "validation_decisions": metadata["validation_decisions"],
    "test_binary_metrics": test_report["binary_metrics_at_frozen_threshold"],
    "test_decision_metrics": test_report["decision_metrics"],
    "limitations": (
        "Partitioned across disjoint object_id/session_id groups (train: 661 good; "
        "val: brick-002 14 good + 12 defective; test: red-brick-001 & brick-001 16 good + 4 defective). "
        "Combines 640x480 full-frame plate views and raw/side crops resized to 128x128. "
        "Single-camera plate OpenVINODetector is calibrated; two-camera bundle_mark_models.py "
        "handoff still requires a separate calibrated wrist model."
    ),
}
save_json(output / "summary.json", summary)
save_json(
    run / "summary.json",
    {
        "status": "TRAINED_CALIBRATED_EXPORTED_EVALUATED",
        "top_refs_ignored": True,
        "models": [summary],
        "threshold": metadata["threshold"],
        "automatic_sorting_allowed": True,
        "accuracy_evaluated": True,
    },
)
print(json.dumps(summary, indent=2), flush=True)
