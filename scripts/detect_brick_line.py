"""Locate the operator-confirmed line in saved brick-002 top RGB captures."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

from brickcell.inspection.line_reference import dark_residual, localize
from brickcell.util import save_json, sha256, versions

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def blue_mask(rgb):
    r, g, b = cv2.split(rgb.astype(float))
    return (b > 1.3 * r) & (g > 1.3 * r) & (g > 60)


def frames(folder):
    record = json.loads((folder / "capture-record.json").read_text())
    paths = [folder / r["files"]["rgb"] for r in record["snapshots"] if r["camera"] == "top"]
    return paths, [np.asarray(Image.open(path).convert("RGB"))[220:272, 225:272] for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clean_paths, clean = frames(args.baseline)
    changed_paths, changed = frames(args.changed)
    reference = np.median(clean[:6], axis=0).astype(np.uint8)
    mask = blue_mask(reference)
    noise = []
    for image in clean[6:]:
        residual, support, _, _ = dark_residual(reference, image, mask, blue_mask(image))
        noise.extend(residual[support].tolist())
    threshold = max(12, float(np.percentile(noise, 99.5) + 4))
    marked = np.median(changed, axis=0).astype(np.uint8)
    result, residual, aligned = localize(reference, marked, mask, blue_mask(marked), threshold)
    counts = []
    for image in changed:
        row, _, _ = localize(reference, image, mask, blue_mask(image), threshold)
        counts.append(len(row["boxes"]))
    overlay = cv2.resize(aligned, (470, 520), interpolation=cv2.INTER_NEAREST)
    for candidate in result["boxes"]:
        x0, y0, x1, y1 = candidate["box"]
        cv2.rectangle(overlay, (x0 * 10, y0 * 10), (x1 * 10, y1 * 10), (255, 60, 40), 2)
        candidate["full_image_box"] = [x0 + 225, y0 + 220, x1 + 225, y1 + 220]
    Image.fromarray(overlay).save(args.output / "line-overlay.png")
    result.update(
        object_id="brick-002",
        camera="top",
        synthetic=False,
        model="reference_comparison_not_PatchCore",
        frames_with_candidate=sum(n > 0 for n in counts),
        changed_frames=len(changed),
        scope="Same-object controlled reference demonstration; not held-out accuracy or a production threshold",
        source_rgb_hashes={str(p): sha256(p) for p in clean_paths + changed_paths},
        versions=versions(),
    )
    save_json(args.output / "line-result.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(reference)
    axes[0].set_title("Clean reference")
    axes[1].imshow(aligned)
    axes[1].set_title("Changed RGB, aligned")
    axes[2].imshow(overlay)
    axes[2].set_title("Localized dark mark")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "line-comparison.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["versions", "source_rgb_hashes"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
