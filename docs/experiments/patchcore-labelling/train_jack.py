"""Reproducible clean-only reference fit; no threshold or held-out metrics."""

import json
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

root = Path(__file__).resolve().parent
source = root.parent / "imported-datasets/jack-anomaly-data-positive/positive"
samples, provenance = [], []
for i, line in enumerate((source / "index.jsonl").read_text().splitlines()):
    row = json.loads(line)
    assert row["condition"] == row["label"] == "positive"
    assert row["synthetic"] is False
    assert row["camera"] == "plate" and row["camera_serial"] == "243622060450"
    path = (source / row["path"]).resolve()
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
    provenance.append({**row, "sha256": sha256(path), "image_id": samples[-1].image_id})
assert len(samples) == 570
manifest = Manifest(root / "manifest.json", samples)
save_json(
    manifest.path,
    {"schema_version": 1, "scope": "CLEAN_REFERENCE_ONLY", "samples": [asdict(s) for s in samples]},
)
audit = validate(manifest, require_partitions=False)
save_json(root / "dataset-audit.json", audit)
assert not audit["errors"], audit
save_json(
    root / "provenance.json",
    {
        "source_commit": "0a6b9d3",
        "source_index_sha256": sha256(source / "index.jsonl"),
        "samples": provenance,
    },
)
config = ModelConfig(
    manifest.path,
    root / "model",
    sensor_contract={"serial": "243622060450", "crop": [0, 0, 640, 480], "modality": "rgb"},
)
config.output_dir.mkdir(exist_ok=False)
save_json(root / "training-config.json", asdict(config))
save_json(root / "versions.json", versions())
torch.set_num_threads(config.torch_threads)
seed_everything(config.seed, workers=True)
started = time.perf_counter()
model = build_model(config)
engine = Engine(
    accelerator="cpu",
    devices=1,
    max_epochs=1,
    logger=False,
    enable_progress_bar=False,
    default_root_dir=config.output_dir / "engine",
    limit_val_batches=0,
    num_sanity_val_steps=0,
)
engine.fit(
    model=model,
    train_dataloaders=loader(
        manifest, "train", config.image_size, config.batch_size, config.num_workers
    ),
)
checkpoint = config.output_dir / "reference.ckpt"
engine.trainer.save_checkpoint(checkpoint)
# Restore and score one fitting image solely to verify checkpoint usability.
restored = type(model).load_from_checkpoint(str(checkpoint), map_location="cpu", pre_trained=False)
restored.cpu().eval()
batch = next(iter(loader(manifest, "train", config.image_size, 1, 0)))
with torch.inference_mode():
    prediction = restored(batch.image)
assert torch.isfinite(prediction.pred_score).all()
save_json(
    root / "summary.json",
    {
        "status": "TRAINED",
        "scope": "CLEAN_REFERENCE_ONLY_UNCALIBRATED",
        "architecture": "Anomalib 2.7.0 PatchCore",
        "device": "cpu",
        "reference_count": len(samples),
        "objects": sorted({s.object_id for s in samples}),
        "sessions": sorted({s.session_id for s in samples}),
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
        "missing_requested_collection": "Earlier approximately 650 images: location/labels pending",
        "limitations": "One capture session; full-frame RGB; no independent calibration/test data.",
    },
)
print(json.dumps(json.loads((root / "summary.json").read_text()), indent=2))
