"""Fit camera-specific clean reference pilots on Intel; never calibrate or sort."""

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
inventory = json.loads((run / "inventory.json").read_text())
assert len(inventory["images"]) == 650
hashes = set()
for row in inventory["images"]:
    assert sha256(run / row["path"]) == row["sha256"]
    assert row["sha256"] not in hashes
    hashes.add(row["sha256"])

jack = run.parent / "imported-datasets/jack-anomaly-data-positive/positive"
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

save_json(run / "versions.json", versions())
save_json(
    run / "input-audit.json",
    {
        "earlier_count": 650,
        "jack_count": 570,
        "unique_hashes": len(hashes),
        "inventory_sha256": sha256(run / "inventory.json"),
        "script_sha256": sha256(Path(__file__)),
        "excluded_count": sum(
            r["disposition"] != "accepted_clean_reference_only" for r in inventory["images"]
        ),
        "scope": "All source bytes verified; original training eligibility and labels unchanged",
    },
)
torch.set_num_threads(4)
summaries = []
for camera, serial, expected_count in [("top", "350122060653", 12), ("plate", "243622060450", 672)]:
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
                path=str((run / row["path"]).resolve()),
                label="good",
                split="train",
                object_id=m["object_id"],
                session_id=m["session_id"],
                view_id=m.get("view_id", camera),
            )
        )
        provenance.append(row)
    earlier_count = len(samples)
    if camera == "plate":
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
    assert len(samples) == expected_count
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
        f"Fitting {camera}: {earlier_count} earlier + {len(samples) - earlier_count} Jack",
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
    restored = type(model).load_from_checkpoint(
        str(checkpoint), map_location="cpu", pre_trained=False
    )
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
        "jack_reference_count": len(samples) - earlier_count,
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
        "limitations": "Full-frame clean reference pilot. Earlier views are plate-resting; "
        "Jack views are arm-held. No independent calibration/test data. "
        "Original formal-training eligibility remains unchanged.",
    }
    save_json(output / "summary.json", summary)
    summaries.append(summary)
    print(json.dumps(summary), flush=True)
    del engine, model, restored
save_json(
    run / "summary.json",
    {
        "status": "TRAINED",
        "source_images_gathered": 1220,
        "earlier_images_gathered": 650,
        "earlier_clean_references_fitted": 114,
        "earlier_images_excluded": 536,
        "models": summaries,
        "threshold": None,
        "automatic_sorting_allowed": False,
    },
)
