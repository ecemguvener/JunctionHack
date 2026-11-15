"""Fit plate-only clean reference pilot on Intel (ignoring top refs); never calibrate or sort."""

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
from anomalib.engine import Engine
from lightning import seed_everything

from brickcell.data.manifest import Manifest, Sample, validate
from brickcell.detector.config import ModelConfig
from brickcell.detector.dataset import loader
from brickcell.detector.workflow import build_model
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

raw_good_dir = Path.home() / "brickcell-captures/raw/good"
reports_dir = Path.home() / "brickcell-captures/reports"
raw_good_side_rows = []
for png_path in sorted(raw_good_dir.rglob("*.png")):
    rel = png_path.relative_to(raw_good_dir)
    if len(rel.parts) != 4 or rel.parts[2] != "side":
        continue
    session_id, object_id, camera, filename = rel.parts
    report_path = reports_dir / f"{png_path.stem}.json"
    assert report_path.exists(), f"Missing report for {png_path}"
    report = json.loads(report_path.read_text())
    assert report.get("human_label") == "good", f"Non-good label in {report_path}"
    assert report.get("synthetic") is False, f"Synthetic flag in {report_path}"
    digest = sha256(png_path)
    assert digest not in hashes, f"Duplicate hash for {png_path}"
    hashes.add(digest)
    raw_good_side_rows.append(
        {
            "image_id": f"rawgood-{png_path.stem[:12]}",
            "path": str(png_path.resolve()),
            "object_id": object_id,
            "session_id": session_id,
            "camera": camera,
            "sha256": digest,
            "report_path": str(report_path.resolve()),
            "crop": report.get("captures", {}).get("side", {}).get("crop"),
        }
    )

assert len(raw_good_side_rows) == 18, (
    f"Expected 18 raw/good/side PNGs, got {len(raw_good_side_rows)}"
)

save_json(run / "versions.json", versions())
save_json(
    run / "input-audit.json",
    {
        "earlier_inventory_count": 650,
        "earlier_plate_count": 102,
        "earlier_top_ignored_count": 12,
        "jack_plate_count": 570,
        "raw_good_side_count": len(raw_good_side_rows),
        "total_plate_fit_count": 102 + 570 + len(raw_good_side_rows),
        "unique_hashes_checked": len(hashes),
        "inventory_sha256": sha256(combined_run / "inventory.json"),
        "script_sha256": sha256(Path(__file__)),
        "scope": "Plate/side clean references only; top refs ignored; all source bytes verified",
    },
)

torch.set_num_threads(4)
camera, serial = "plate", "243622060450"
output = run / camera
output.mkdir(exist_ok=False)
samples, provenance = [], []

for row in inventory["images"]:
    m = row["metadata"]
    if row["disposition"] != "accepted_clean_reference_only" or m["camera"] != camera:
        continue
    assert m["human_label"] == "good" and m["baseline_accepted"] is True
    assert m["serial"] == serial and m["synthetic"] is False
    assert row["dimensions"] == [640, 480]
    samples.append(
        Sample(
            image_id=row["image_id"],
            path=str((combined_run / row["path"]).resolve()),
            label="good",
            split="train",
            object_id=m["object_id"],
            session_id=m["session_id"],
            view_id=m.get("view_id", camera),
        )
    )
    provenance.append(row)

earlier_count = len(samples)
assert earlier_count == 102

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
        )
    )
    provenance.append({**row, "image_id": samples[-1].image_id})

jack_count = len(samples) - earlier_count
assert jack_count == 570

for row in raw_good_side_rows:
    samples.append(
        Sample(
            image_id=row["image_id"],
            path=row["path"],
            label="good",
            split="train",
            object_id=row["object_id"],
            session_id=row["session_id"],
            view_id="side",
        )
    )
    provenance.append(row)

raw_good_count = len(samples) - earlier_count - jack_count
assert len(samples) == 690

manifest = Manifest(output / "manifest.json", samples)
save_json(
    manifest.path,
    {
        "schema_version": 1,
        "scope": "CLEAN_REFERENCE_ONLY",
        "samples": [asdict(s) for s in samples],
    },
)
audit = validate(manifest, require_partitions=False)
save_json(output / "dataset-audit.json", audit)
assert not audit["errors"], audit
save_json(output / "provenance.json", provenance)

config = ModelConfig(
    manifest.path,
    output,
    sensor_contract={"serial": serial, "crop": [0, 0, 640, 480], "modality": "rgb"},
)
save_json(output / "training-config.json", asdict(config))
seed_everything(config.seed, workers=True)
started = time.perf_counter()
print(
    f"Fitting {camera}: {earlier_count} earlier + {jack_count} Jack + {raw_good_count} raw/good/side",
    flush=True,
)
model = build_model(config)
engine = Engine(
    accelerator="cpu",
    devices=1,
    max_epochs=1,
    logger=False,
    enable_progress_bar=False,
    default_root_dir=output / "engine",
    limit_val_batches=0,
    num_sanity_val_steps=0,
)
engine.fit(
    model=model,
    train_dataloaders=loader(
        manifest, "train", config.image_size, config.batch_size, config.num_workers
    ),
)
checkpoint = output / "reference.ckpt"
engine.trainer.save_checkpoint(checkpoint)
restored = type(model).load_from_checkpoint(str(checkpoint), map_location="cpu", pre_trained=False)
restored.cpu().eval()
batch = next(iter(loader(manifest, "train", config.image_size, 1, 0)))
with torch.inference_mode():
    prediction = restored(batch.image)
assert torch.isfinite(prediction.pred_score).all()

summary = {
    "status": "TRAINED",
    "camera": camera,
    "serial": serial,
    "scope": "CLEAN_REFERENCE_ONLY_UNCALIBRATED",
    "device": "cpu",
    "host": platform.node(),
    "architecture": "Anomalib 2.7.0 PatchCore",
    "reference_count": len(samples),
    "earlier_reference_count": earlier_count,
    "jack_reference_count": jack_count,
    "raw_good_side_count": raw_good_count,
    "top_refs_ignored": True,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "manifest_sha256": sha256(manifest.path),
    "training_seconds": time.perf_counter() - started,
    "memory_bank_shape": list(restored.model.memory_bank.shape),
    "restore_smoke_raw_distance": float(prediction.pred_score.item()),
    "threshold": None,
    "automatic_sorting_allowed": False,
    "accuracy_evaluated": False,
    "validation_count": 0,
    "test_count": 0,
    "limitations": "Clean reference pilot ignoring top refs. Combines 102 earlier full-frame "
    "plate-resting views, 570 Jack arm-held views, and 18 cropped raw/good/side views. "
    "No independent calibration/test data.",
}
save_json(output / "summary.json", summary)
save_json(
    run / "summary.json",
    {
        "status": "TRAINED",
        "top_refs_ignored": True,
        "models": [summary],
        "threshold": None,
        "automatic_sorting_allowed": False,
    },
)
print(json.dumps(summary), flush=True)
