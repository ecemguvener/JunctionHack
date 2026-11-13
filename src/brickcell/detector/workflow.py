"""Anomalib 2.7.0 training, external validation calibration, export and parity."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from brickcell.data.manifest import Manifest, load_manifest, require_valid
from brickcell.detector.config import ModelConfig
from brickcell.inspection.images import input_tensor, read_rgb
from brickcell.inspection.threshold import calibrate
from brickcell.metrics.report import classification, latency
from brickcell.util import save_json, sha256, versions


@contextlib.contextmanager
def run_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    logging.getLogger().addHandler(handler)
    with (
        path.open("a", buffering=1) as stream,
        contextlib.redirect_stdout(stream),
        contextlib.redirect_stderr(stream),
    ):
        try:
            yield
        except Exception:
            logging.exception("Workflow failed")
            raise
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()


def build_model(config: ModelConfig, pretrained: bool | None = None):
    from anomalib.models import Patchcore

    return Patchcore(
        backbone=config.backbone,
        layers=config.layers,
        pre_trained=config.pre_trained if pretrained is None else pretrained,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        num_neighbors=config.num_neighbors,
        pre_processor=Patchcore.configure_pre_processor(image_size=config.image_size),
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )


def load_checkpoint(config: ModelConfig, checkpoint: Path | None = None):
    from anomalib.models import Patchcore

    metadata = json.loads((config.output_dir / "metadata.json").read_text())
    checkpoint = checkpoint or config.output_dir / "model.ckpt"
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from calibrated model")
    for field in ("image_size", "backbone", "layers", "num_neighbors"):
        expected = metadata["config"][field]
        actual = getattr(config, field)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            raise ValueError(f"Configuration {field} differs from trained model")
    # Override pre_trained to avoid any network access when restoring learned weights.
    model = Patchcore.load_from_checkpoint(str(checkpoint), map_location="cpu", pre_trained=False)
    return model.cpu().eval()


def torch_predict(model, rgb: np.ndarray, size: tuple[int, int]):
    import torch

    tensor = torch.from_numpy(input_tensor(rgb, size))
    start = time.perf_counter()
    with torch.inference_mode():
        result = model(tensor)
    return float(result.pred_score.item()), (time.perf_counter() - start) * 1000


def verify_dataset_snapshot(config: ModelConfig, manifest: Manifest) -> None:
    """Detect changed image bytes even if the JSON manifest itself is unchanged."""
    provenance = json.loads((config.output_dir / "data_provenance.json").read_text())
    recorded = {row["image_id"]: row for row in provenance["images"]}
    if set(recorded) != {sample.image_id for sample in manifest.samples}:
        raise ValueError("Dataset membership changed since training")
    for sample in manifest.samples:
        row = recorded[sample.image_id]
        if row["split"] != sample.split or row["sha256"] != sha256(manifest.image_path(sample)):
            raise ValueError(f"Dataset changed since training: {sample.image_id}")


def train(config: ModelConfig) -> dict:
    import torch
    from anomalib.engine import Engine
    from lightning import seed_everything

    from brickcell.detector.dataset import loader

    manifest = load_manifest(config.manifest)
    report = require_valid(manifest)
    if (config.output_dir / "model.ckpt").exists():
        raise ValueError(
            "Output already contains a checkpoint; use a new output_dir to preserve provenance"
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.output_dir / "training_config.json", asdict(config))
    save_json(config.output_dir / "package_versions.json", versions())
    save_json(config.output_dir / "dataset_validation.json", report)
    save_json(config.output_dir / "split_manifest.json", json.loads(config.manifest.read_text()))
    save_json(
        config.output_dir / "data_provenance.json",
        {
            "source_manifest": str(config.manifest),
            "images": [
                {
                    "image_id": s.image_id,
                    "split": s.split,
                    "sha256": sha256(manifest.image_path(s)),
                    "object_id": s.object_id,
                    "session_id": s.session_id,
                }
                for s in manifest.samples
            ],
        },
    )
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed, workers=True)
    started = time.perf_counter()
    with run_log(config.output_dir / "training.log"):
        model = build_model(config)
        engine = Engine(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_progress_bar=False,
            deterministic=True,
            default_root_dir=config.output_dir / "engine",
        )
        train_loader = loader(
            manifest, "train", config.image_size, config.batch_size, config.num_workers
        )
        val_loader = loader(
            manifest, "val", config.image_size, config.batch_size, config.num_workers
        )
        engine.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        engine.trainer.save_checkpoint(config.output_dir / "model.ckpt")
        model.eval()
        rows = []
        for sample in manifest.partition("val"):
            score, elapsed = torch_predict(
                model, read_rgb(manifest.image_path(sample)), config.image_size
            )
            rows.append(
                {
                    "image_id": sample.image_id,
                    "score": score,
                    "label": int(sample.label == "defective"),
                    "latency_ms": elapsed,
                }
            )
        scores, labels = [r["score"] for r in rows], [r["label"] for r in rows]
        threshold = calibrate(
            scores, labels, report["manifest_sha256"], config.uncertainty_fraction
        )
        checkpoint_hash = sha256(config.output_dir / "model.ckpt")
        metadata = {
            "schema_version": 1,
            "model_version": checkpoint_hash[:16],
            "checkpoint_sha256": checkpoint_hash,
            "config": asdict(config),
            "input_size": config.image_size,
            "threshold": asdict(threshold),
            "score_type": threshold.score_type,
            "preprocessing": "EXIF transpose; RGB; Pillow bilinear resize; NCHW float32 /255; embedded ImageNet normalization once",
            "synthetic": any(s.synthetic for s in manifest.samples),
            "backbone": config.backbone,
            "pretrained": config.pre_trained,
            "backbone_pretrained_config": getattr(
                model.model.feature_extractor.feature_extractor, "pretrained_cfg", {}
            ),
            "training_seconds": time.perf_counter() - started,
            "validation_metrics": classification(labels, scores, threshold.value),
            "validation_latency": latency([r["latency_ms"] for r in rows]),
        }
        save_json(config.output_dir / "metadata.json", metadata)
        save_json(config.output_dir / "validation_predictions.json", rows)
    return metadata


def export(config: ModelConfig, checkpoint: Path | None = None) -> dict:
    import openvino as ov
    import torch
    from anomalib.engine import Engine

    torch.set_num_threads(config.torch_threads)
    checkpoint = checkpoint or config.output_dir / "model.ckpt"
    metadata = json.loads((config.output_dir / "metadata.json").read_text())
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match calibrated model metadata")
    with run_log(config.output_dir / "export.log"):
        model = load_checkpoint(config, checkpoint)
        engine = Engine(
            accelerator="cpu",
            devices=1,
            logger=False,
            default_root_dir=config.output_dir / "export_engine",
        )
        path = engine.export(
            model=model,
            export_type="openvino",
            export_root=config.output_dir / "export",
            model_file_name="model",
            input_size=config.image_size,
            compression_type=None,
            onnx_kwargs={"dynamic_axes": {}, "dynamic_shapes": None, "opset_version": 18},
        )
        if path is None:
            raise RuntimeError("Anomalib export did not return a model path")
        import shutil

        final = config.output_dir / "model.xml"
        shutil.copy2(path, final)
        shutil.copy2(Path(path).with_suffix(".bin"), final.with_suffix(".bin"))
        ir = ov.Core().read_model(str(final))
        shape = list(ir.input(0).shape)
        if shape != [1, 3, *config.image_size]:
            raise ValueError(f"Expected static NCHW input, found {shape}")
        metadata.update(
            model_sha256=sha256(final),
            weights_sha256=sha256(final.with_suffix(".bin")),
            export_device="CPU",
            compression="none",
            input_shape=shape,
            output_shapes={p.get_any_name(): str(p.partial_shape) for p in ir.outputs},
            export_versions=versions(),
        )
        save_json(final.with_suffix(".json"), metadata)
    return {
        "model": str(final),
        "weights": str(final.with_suffix(".bin")),
        "metadata": str(final.with_suffix(".json")),
        "input_shape": shape,
        "output_shapes": metadata["output_shapes"],
    }


def parity(config: ModelConfig, tolerance: float = 0.001) -> dict:
    import torch

    from brickcell.detector.runtime import OpenVINODetector

    torch.set_num_threads(config.torch_threads)
    manifest = load_manifest(config.manifest)
    require_valid(manifest)
    verify_dataset_snapshot(config, manifest)
    model = load_checkpoint(config)
    detector = OpenVINODetector(config.output_dir / "model.xml")
    if detector.threshold is None:
        raise ValueError("Parity labels require a calibrated threshold")
    rows = []
    # Use validation only: parity debugging must not become test-set model selection.
    for sample in manifest.partition("val"):
        rgb = read_rgb(manifest.image_path(sample))
        score, _ = torch_predict(model, rgb, config.image_size)
        other = detector.predict(rgb).score
        label = detector.threshold.decide(score)
        ov_label = detector.threshold.decide(other)
        rows.append(
            {
                "image_id": sample.image_id,
                "torch_score": score,
                "openvino_score": other,
                "absolute_difference": abs(score - other),
                "torch_decision": label,
                "openvino_decision": ov_label,
                "label_match": label == ov_label,
            }
        )
    result = {
        "partition": "val",
        "absolute_tolerance": tolerance,
        "passed": all(r["absolute_difference"] <= tolerance and r["label_match"] for r in rows),
        "max_absolute_difference": max(r["absolute_difference"] for r in rows),
        "label_mismatches": sum(not r["label_match"] for r in rows),
        "samples": rows,
    }
    save_json(config.output_dir / "parity.json", result)
    return result


def evaluate(config: ModelConfig) -> dict:
    from brickcell.detector.runtime import OpenVINODetector

    manifest = load_manifest(config.manifest)
    require_valid(manifest)
    verify_dataset_snapshot(config, manifest)
    detector = OpenVINODetector(config.output_dir / "model.xml")
    if detector.threshold is None:
        raise ValueError("Evaluation requires a frozen validation threshold")
    if sha256(config.manifest) != detector.threshold.calibration_manifest_sha256:
        raise ValueError("Manifest changed since calibration; preserve the frozen split manifest")
    rows = []
    for sample in manifest.partition("test"):
        prediction = detector.predict(read_rgb(manifest.image_path(sample)))
        rows.append(
            {
                "image_id": sample.image_id,
                "score": prediction.score,
                "label": int(sample.label == "defective"),
                "latency_ms": prediction.latency_ms,
                "decision": detector.threshold.decide(prediction.score),
            }
        )
    result = {
        "partition": "held_out_test",
        "model_version": detector.model_version,
        "threshold_version": detector.threshold.version,
        "synthetic": any(s.synthetic for s in manifest.partition("test")),
        "scope": "Detector classification, before view-quality gating; no physical robot measurements",
        "binary_metrics_at_frozen_threshold": classification(
            [r["label"] for r in rows], [r["score"] for r in rows], detector.threshold.value
        ),
        "uncertain_count": sum(r["decision"] == "REINSPECT" for r in rows),
        "latency": latency([r["latency_ms"] for r in rows]),
        "samples": rows,
    }
    save_json(config.output_dir / "test_report.json", result)
    return result
