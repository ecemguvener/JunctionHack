"""Extract red brick regions and merge with a provisional review transform."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    args = parser.parse_args()
    tf = json.loads(args.transform.read_text())
    rotation = np.asarray(tf["top_from_plate_R"])
    translation = np.asarray(tf["top_from_plate_t_m"])
    clouds, counts = [], {}
    for name, roi in [("top", (229, 219, 282, 272)), ("plate", (362, 70, 431, 140))]:
        packet = json.loads((args.directory / f"{name}-points.json").read_text())
        if packet["serial"] != tf["sdk_serials"][name]:
            raise ValueError("Transform camera mismatch")
        cloud = np.asarray(packet["points"], dtype=float)
        i = packet["intrinsics"]
        u = cloud[:, 0] / cloud[:, 2] * i["fx"] + i["ppx"]
        v = cloud[:, 1] / cloud[:, 2] * i["fy"] + i["ppy"]
        x0, y0, x1, y1 = roi
        red = (cloud[:, 3] > 1.35 * cloud[:, 4]) & (cloud[:, 3] > 1.35 * cloud[:, 5])
        keep = red & (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
        cloud = cloud[keep]
        counts[name] = len(cloud)
        if name == "plate":
            cloud[:, :3] = cloud[:, :3] @ rotation.T + translation
        clouds.append(cloud)
    cloud = np.vstack(clouds)
    if not len(cloud):
        raise ValueError("No brick points found")
    with (args.directory / "brick-001-rough-merged.ply").open("w") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(cloud)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        np.savetxt(f, cloud, fmt=["%.5f"] * 3 + ["%d"] * 3)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    for name, c in zip(["top", "plate"], clouds):
        ax.scatter(c[:, 0] * 1000, c[:, 1] * 1000, c[:, 2] * 1000, s=8, label=name)
    ax.set(
        xlabel="X (mm)",
        ylabel="Y (mm)",
        zlabel="Z (mm)",
        title="Brick-001: rough merged depth views",
    )
    ax.set_box_aspect(np.maximum(np.ptp(cloud[:, :3], axis=0), 0.001))
    ax.legend()
    fig.savefig(args.directory / "merged-preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    report = dict(
        object_id="brick-001",
        synthetic=False,
        provisional=True,
        points=counts,
        limitations=[
            "Capture-specific red color and pixel ROI selection",
            "Coarse alignment; not defect metrology",
            "Partial visible surfaces, not a watertight reconstruction",
        ],
    )
    (args.directory / "merge-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
