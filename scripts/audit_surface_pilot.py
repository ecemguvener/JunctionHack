"""Controlled synthetic perturbations of a reference image, never real accuracy.

Uses the pilot's frozen foreground contract. No fitting, threshold selection or
held-out test evaluation occurs. Reports mark sensitivity and nuisance responses.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.models import Patchcore
from PIL import Image

from brickcell.inspection.block_surface import (
    SurfaceViewError,
    isolate_surface,
    surface_input,
    surface_score,
)
from brickcell.inspection.images import read_rgb
from brickcell.util import save_json, sha256, versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory to preserve evidence")
    metadata = json.loads((args.pilot / "pilot.json").read_text())
    profile = metadata.get("surface_profile")
    if not profile:
        raise ValueError("Requires a pilot trained with a block-surface contract")
    checkpoint = args.pilot / "reference.ckpt"
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    rgb = read_rgb(args.image)
    _, mask = isolate_surface(rgb, profile)
    size = tuple(metadata["image_size"])
    baseline_tensor, _ = surface_input(rgb, size, profile)
    variants = [("original", "reference", rgb)]
    for name, color in (("dark", (0, 0, 0)), ("white", (255, 255, 255)), ("warm", (160, 100, 50))):
        edited = rgb.copy()
        edited[~mask] = color
        variants.append(("background_" + name, "synthetic_background", edited))
    for name, gain in (("dim", 0.8), ("bright", 1.2)):
        edited = np.clip(np.rint(rgb.astype(float) * gain), 0, 255).astype(np.uint8)
        variants.append(("exposure_" + name, "synthetic_global_exposure", edited))
    # A smooth illumination gradient is intentionally included as a hard case;
    # global exposure normalization does not establish local-shadow invariance.
    gradient = np.linspace(0.65, 1.0, rgb.shape[1])[None, :, None]
    edited = np.clip(np.rint(rgb.astype(float) * gradient), 0, 255).astype(np.uint8)
    variants.append(("smooth_shadow", "synthetic_local_illumination", edited))
    interior = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), dtype=np.uint8)).astype(bool)
    y, x = np.unravel_index(
        np.argmax(cv2.distanceTransform(interior.astype(np.uint8), cv2.DIST_L2, 5)), mask.shape
    )
    if not interior[y, x]:
        raise ValueError("Not enough interior surface for a synthetic mark probe")
    for width in (1, 2, 4):
        line = np.zeros(mask.shape, np.uint8)
        cv2.line(line, (max(0, x - 6), y), (min(mask.shape[1] - 1, x + 6), y), 1, width)
        line = line.astype(bool) & interior
        edited = rgb.copy()
        edited[line] = 0
        variants.append((f"black_line_{width}px", "synthetic_mark", edited))
    torch.set_num_threads(4)
    model = (
        Patchcore.load_from_checkpoint(str(checkpoint), map_location="cpu", pre_trained=False)
        .cpu()
        .eval()
    )
    rows = []
    for name, kind, pixels in variants:
        path = args.output / (name + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        row = dict(name=name, kind=kind, synthetic=kind != "reference", image_sha256=sha256(path))
        try:
            tensor, surface = surface_input(pixels, size, profile)
            with torch.inference_mode():
                prediction = model(torch.from_numpy(tensor))
            distance, _ = surface_score(
                float(prediction.pred_score.item()),
                prediction.anomaly_map.squeeze().cpu().numpy(),
                surface,
            )
            row.update(
                raw_surface_distance=distance,
                input_max_absolute_difference=float(np.abs(tensor - baseline_tensor).max()),
                decision="UNCALIBRATED",
            )
        except SurfaceViewError as exc:
            row.update(decision="REINSPECT", reason=str(exc), raw_surface_distance=None)
        rows.append(row)
    baseline = rows[0]["raw_surface_distance"]
    for row in rows:
        row["score_change_from_reference"] = (
            row["raw_surface_distance"] - baseline
            if row["raw_surface_distance"] is not None
            else None
        )
    report = dict(
        scope="SYNTHETIC_PERTURBATION_DIAGNOSTIC_NOT_REAL_DEFECT_ACCURACY",
        source_image=str(args.image.resolve()),
        source_sha256=sha256(args.image),
        checkpoint_sha256=metadata["checkpoint_sha256"],
        surface_profile=profile,
        threshold=None,
        automatic_sorting_allowed=False,
        test_set_used=False,
        versions=versions(),
        variants=rows,
    )
    save_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / "report.json"), "variants": rows}, indent=2))


if __name__ == "__main__":
    main()
