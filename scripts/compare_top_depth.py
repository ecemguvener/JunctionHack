"""Diagnostic RGB/depth comparison for a user-identified top-facing change.

No trained model, geometric registration or defect threshold is applied.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

from brickcell.util import save_json, sha256

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def load(folder, roi):
    record = json.loads((folder / "capture-record.json").read_text())
    rows = [r for r in record["snapshots"] if r["camera"] == "top"]
    if len(rows) < 3:
        raise ValueError("Need repeated top snapshots")
    x0, y0, x1, y1 = roi
    rgbs = np.array(
        [np.asarray(Image.open(folder / r["files"]["rgb"]))[y0:y1, x0:x1] for r in rows]
    )
    depths = np.array(
        [
            np.asarray(Image.open(folder / r["files"]["depth"]))[y0:y1, x0:x1].astype(float)
            * r["depth_scale_m"]
            * 1000
            for r in rows
        ]
    )
    valid = (depths >= 150) & (depths <= 1500)
    masked = np.ma.array(depths, mask=~valid)
    depth = np.ma.median(masked, axis=0).filled(np.nan)
    rgb = np.median(rgbs, axis=0).astype(np.uint8)
    r, g, b = cv2.split(rgb.astype(float))
    blue = ((b > 1.3 * r) & (g > 1.3 * r) & (g > 60)).astype(np.uint8)
    return rgb, depth, blue, valid.sum(0), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi", type=int, nargs=4, default=[225, 220, 272, 272])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    a, b = [load(p, args.roi) for p in [args.baseline, args.changed]]
    if a[4][0]["serial"] != b[4][0]["serial"]:
        raise ValueError("Camera identity changed")
    mask = (cv2.erode(a[2], np.ones((3, 3), np.uint8)) > 0) & (a[3] >= 10) & (b[3] >= 10)
    if mask.sum() < 10:
        raise ValueError("Insufficient supported brick interior")
    delta = a[1] - b[1]
    rgb_delta = abs(a[0].astype(float) - b[0].astype(float)).mean(2)
    centroids = []
    for item in [a, b]:
        y, x = np.where(item[2])
        centroids.append([float(x.mean()), float(y.mean())])
    report = dict(
        object_id="brick-002",
        synthetic=False,
        model_used=False,
        decision="REINSPECT",
        operator_defect_location="surface facing top camera",
        roi=args.roi,
        supported_pixels=int(mask.sum()),
        median_signed_depth_change_mm=float(np.median(delta[mask])),
        p95_absolute_depth_change_mm=float(np.percentile(abs(delta[mask]), 95)),
        median_rgb_absolute_change_0_255=float(np.median(rgb_delta[mask])),
        blue_centroid_shift_pixels=(np.array(centroids[1]) - centroids[0]).tolist(),
        source_record_hashes=[
            sha256(p / "capture-record.json") for p in [args.baseline, args.changed]
        ],
        limitations=[
            "Direct pixels; subpixel object/framing shift remains",
            "Blue reference mask is a heuristic and can omit some defect regions",
            "No independent threshold, registration or trained detection",
            "Plate view excluded due to larger framing change",
        ],
    )
    save_json(args.output / "comparison-report.json", report)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(a[0])
    axes[0].set_title("Clean · top RGB")
    axes[1].imshow(b[0])
    axes[1].set_title("Added change · top RGB")
    image = axes[2].imshow(np.where(mask, delta, np.nan), cmap="coolwarm", vmin=-6, vmax=6)
    axes[2].set_title("Clean − changed depth (mm)")
    fig.colorbar(image, ax=axes[2], shrink=0.7)
    axes[3].imshow(mask, cmap="gray")
    axes[3].set_title("Supported interior pixels")
    for axis in axes:
        axis.axis("off")
    fig.suptitle("Top-facing change: measured comparison, NOT a trained defect result")
    fig.tight_layout()
    fig.savefig(args.output / "top-comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
