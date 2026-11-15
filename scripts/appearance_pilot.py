"""Reference-only PatchCore pilot. Produces distances/maps, never calibrated decisions.

One object's before/after comparison is a diagnostic, not held-out accuracy.
Use the normal train/val/test workflow for any calibrated inspection model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brickcell.util import save_json, sha256, versions


def fit(
    references: Path,
    output: Path,
    object_id: str,
    session_id: str,
    crop: list[int],
    camera: str = "side",
    image_size: int = 128,
    surface_profile: dict | None = None,
    layers: list[str] | None = None,
    foreground_memory: bool = False,
) -> dict:
    import torch
    from anomalib.engine import Engine
    from lightning import seed_everything

    from brickcell.data.manifest import Manifest, Sample
    from brickcell.detector.config import ModelConfig
    from brickcell.detector.dataset import loader
    from brickcell.detector.workflow import build_model

    paths = sorted(
        p.resolve() for p in references.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise ValueError("No clean reference images")
    if output.exists():
        raise ValueError("Choose a new output directory; existing pilots are preserved")
    hashes = [sha256(path) for path in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate reference image bytes")
    config = ModelConfig(
        manifest=references,
        output_dir=output,
        image_size=(image_size, image_size),
        surface_profile=surface_profile,
        layers=layers or ["layer2", "layer3"],
        foreground_memory=foreground_memory,
    )
    manifest = Manifest(
        references / "reference-only.json",
        [
            Sample(
                f"reference-{i}",
                str(path),
                "good",
                "train",
                object_id=object_id,
                session_id=session_id,
                view_id=camera,
            )
            for i, path in enumerate(paths)
        ],
    )
    torch.set_num_threads(4)
    seed_everything(42, workers=True)
    output.mkdir(parents=True)
    save_json(
        output / "references.json",
        {
            "scope": "reference-only pilot; no val/test partitions or accuracy evaluation",
            "object_id": object_id,
            "session_id": session_id,
            "camera": camera,
            "crop": crop,
            "references": [{"path": str(p), "sha256": h} for p, h in zip(paths, hashes)],
        },
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
        train_dataloaders=loader(manifest, "train", config.image_size, 4, 0, surface_profile),
    )
    checkpoint = output / "reference.ckpt"
    engine.trainer.save_checkpoint(checkpoint)
    metadata = {
        "scope": "SAME_OBJECT_REFERENCE_PILOT_NOT_ACCURACY_EVALUATION",
        "synthetic": False,
        "object_id": object_id,
        "session_id": session_id,
        "camera": camera,
        "crop": crop,
        "reference_count": len(paths),
        "checkpoint_sha256": sha256(checkpoint),
        "image_size": list(config.image_size),
        "surface_profile": surface_profile,
        "layers": config.layers,
        "foreground_memory": foreground_memory,
        "foreground_training_counts": getattr(model, "foreground_training_counts", None),
        "score_aggregation": "max_patchcore_map_on_block_mask_v1"
        if surface_profile
        else "patchcore_pred_score",
        "threshold": None,
        "automatic_sorting_allowed": False,
        "references": [{"path": str(p), "sha256": h} for p, h in zip(paths, hashes)],
        "versions": versions(),
    }
    save_json(output / "pilot.json", metadata)
    return {"pilot": str(output.resolve()), "reference_count": len(paths), "threshold": None}


def score(pilot: Path, images: list[Path]) -> dict:
    import torch
    from anomalib.models import Patchcore

    from brickcell.detector.runtime import Prediction, save_map
    from brickcell.inspection.block_surface import surface_input, surface_score
    from brickcell.inspection.images import read_rgb

    metadata = json.loads((pilot / "pilot.json").read_text())
    checkpoint = pilot / "reference.ckpt"
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Pilot checkpoint hash mismatch")
    torch.set_num_threads(4)
    model = Patchcore.load_from_checkpoint(str(checkpoint), map_location="cpu", pre_trained=False)
    model.cpu().eval()
    rows = []
    for path in images:
        rgb = read_rgb(path)
        prepared, mask = surface_input(
            rgb, tuple(metadata["image_size"]), metadata.get("surface_profile")
        )
        tensor = torch.from_numpy(prepared)
        with torch.inference_mode():
            result = model(tensor)
        distance, amap = surface_score(
            float(result.pred_score.item()), result.anomaly_map.squeeze().cpu().numpy(), mask
        )
        prediction = Prediction(distance, 0, amap)
        digest = sha256(path)
        map_path = save_map(prediction, rgb, pilot / "maps" / digest[:20])
        rows.append(
            {
                "image": str(path.resolve()),
                "sha256": digest,
                "raw_patchcore_distance": prediction.score,
                "map": map_path,
                "is_fitting_reference": any(r["sha256"] == digest for r in metadata["references"]),
            }
        )
    report = {
        "scope": metadata["scope"],
        "decision": "UNCALIBRATED",
        "threshold": None,
        "accuracy_evaluated": False,
        "images": rows,
    }
    save_json(pilot / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit")
    fitting.add_argument("--references", type=Path, required=True)
    fitting.add_argument("--output", type=Path, required=True)
    fitting.add_argument("--object-id", required=True)
    fitting.add_argument("--session-id", required=True)
    fitting.add_argument("--crop", nargs=4, type=int, required=True)
    fitting.add_argument("--camera", choices=["plate", "wrist", "side", "top"], default="side")
    fitting.add_argument("--image-size", type=int, default=128)
    fitting.add_argument("--surface-profile", type=Path)
    fitting.add_argument("--foreground-memory", action="store_true")
    fitting.add_argument(
        "--layers", nargs="+", choices=["layer1", "layer2", "layer3"], default=["layer2", "layer3"]
    )
    scoring = commands.add_parser("score")
    scoring.add_argument("--pilot", type=Path, required=True)
    scoring.add_argument("--images", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    result = (
        fit(
            args.references,
            args.output,
            args.object_id,
            args.session_id,
            args.crop,
            args.camera,
            args.image_size,
            json.loads(args.surface_profile.read_text()) if args.surface_profile else None,
            args.layers,
            args.foreground_memory,
        )
        if args.command == "fit"
        else score(args.pilot, args.images)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
