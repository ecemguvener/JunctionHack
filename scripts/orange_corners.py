"""Find color-contrast corner candidates on an irregular orange board.

Candidates are for manual review, not ordered calibration correspondences.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detect(image, polygon):
    b, g, r = cv2.split(image.astype(np.float32))
    contrast = (r - g) / (r + g + b + 1)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
    values = contrast[mask > 0]
    lo, hi = np.percentile(values, [5, 95])
    if hi - lo < 0.005:
        raise ValueError("Insufficient orange color contrast")
    channel = (255 * np.clip((contrast - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    channel = cv2.GaussianBlur(channel, (3, 3), 0)
    corners = cv2.goodFeaturesToTrack(channel, 80, 0.08, 9, mask=mask, blockSize=5)
    points = [] if corners is None else corners.reshape(-1, 2).tolist()
    overlay = image.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(overlay, (round(x), round(y)), 4, (0, 255, 0), 1)
        cv2.putText(
            overlay, str(i), (round(x) + 4, round(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1
        )
    return points, channel, overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    # Explicit polygons are specific to the reviewed 2026-11-14 capture pair.
    polygons = {
        "top": [[145, 197], [258, 152], [367, 300], [236, 359]],
        "plate": [[224, 349], [385, 253], [639, 277], [639, 420]],
    }
    report = {
        "calibrated": False,
        "method": "normalized red-green contrast + Shi-Tomasi",
        "views": {},
    }
    for name, polygon in polygons.items():
        image = cv2.imread(str(args.directory / f"{name}.jpg"))
        if image is None:
            raise ValueError(f"Cannot read {name}")
        points, channel, overlay = detect(image, polygon)
        cv2.imwrite(str(args.directory / f"{name}-orange-contrast.png"), channel)
        cv2.imwrite(str(args.directory / f"{name}-corner-candidates.png"), overlay)
        report["views"][name] = {
            "polygon": polygon,
            "candidate_count": len(points),
            "candidate_pixels": points,
        }
    report["limitations"] = [
        "Unordered candidates may include edge or texture artifacts",
        "No cross-camera matching or transform",
        "Plate view clips the right side of this target",
        "Irregular grid cannot be passed to standard equal-square calibration",
    ]
    (args.directory / "corner-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print({name: data["candidate_count"] for name, data in report["views"].items()})


if __name__ == "__main__":
    main()
