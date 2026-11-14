"""Review a manually selected brick region in two aligned D415 captures.

Uses existing SDK-exported PLY vertices (viewer axes x, -y, -z).
ROI selection is operator-assisted; output is an uncalibrated diagnostic.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def vertices(path):
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Expected SDK binary little-endian PLY")
        count = int(next(x.split()[-1] for x in header if x.startswith("element vertex")))
        dtype = np.dtype(
            [(n, "<f4") for n in "xyz"] + [(n, "u1") for n in ("red", "green", "blue")]
        )
        return np.fromfile(stream, dtype=dtype, count=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--defect", type=Path, required=True)
    parser.add_argument("--roi", type=int, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"synthetic": False, "calibrated": False, "roi": args.roi, "scans": {}}
    x0, y0, x1, y1 = args.roi
    figure = plt.figure(figsize=(10, 5))
    for index, (name, folder) in enumerate((("clean", args.clean), ("defect", args.defect))):
        v = vertices(folder / "plate.ply")
        z = -v["z"]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = 607.7897339 * v["x"] / z + 320.0687866
            w = -606.9682617 * v["y"] / z + 247.1624603
        keep = (z > 0.1) & (z < 2) & (u >= x0) & (u < x1) & (w >= y0) & (w < y1)
        selected = v[keep]
        axis = figure.add_subplot(1, 2, index + 1, projection="3d")
        if len(selected):
            xyz = np.column_stack([selected[n] for n in "xyz"]) * 1000
            rgb = np.column_stack([selected[n] for n in ("red", "green", "blue")]) / 255
            axis.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], c=rgb, s=2)
            axis.set_box_aspect(np.maximum(np.ptp(xyz[:, [0, 2, 1]], axis=0), 1))
        axis.set_title(f"{name}: {len(selected)} ROI points")
        axis.set_xlabel("X (mm)")
        axis.set_ylabel("SDK viewer Z (mm)")
        axis.set_zlabel("SDK viewer Y (mm)")
        path = args.output / f"{name}-brick-roi.ply"
        with path.open("wb") as stream:
            stream.write(
                (
                    "ply\nformat binary_little_endian 1.0\nelement vertex "
                    + str(len(selected))
                    + "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                ).encode()
            )
            selected.tofile(stream)
        Image.open(folder / "plate-color.png").crop(args.roi).save(
            args.output / f"{name}-rgb-roi.png"
        )
        depth = np.asarray(Image.open(folder / "plate-depth16.png"))[y0:y1, x0:x1]
        valid = (depth > 0) & (depth < 2000)
        report["scans"][name] = {
            "points_in_roi": len(selected),
            "valid_depth_fraction": float(valid.mean()),
            "source_ply_sha256": hashlib.sha256((folder / "plate.ply").read_bytes()).hexdigest(),
        }
    report["limitations"] = [
        "Manual image ROI may include table or miss brick edges; review the RGB crops.",
        "Fixed intrinsics copied from this plate camera's 640x480 SDK probe.",
        "No pose registration, noise calibration, trained detector or defect decision.",
    ]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    figure.suptitle("Real depth captures — manual brick region, uncalibrated comparison")
    figure.savefig(args.output / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
