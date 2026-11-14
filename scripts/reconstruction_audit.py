"""Compare original ICP with cleaned merged-camera and plate-only pose fits."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np

from brickcell.reconstruction import clean_points, register

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def load(path):
    with path.open() as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Missing PLY header")
            if line.strip() == "end_header":
                break
        return np.loadtxt(stream)


def save(path, points):
    with path.open("w") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        )
        np.savetxt(stream, points, fmt="%.6f")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/brick-001-reconstruction-audit")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ["rotated", "narrow", "back", "other-side", "merged"]
    reports = {}
    methods = {}
    raw = {}
    for name in names:
        folder = Path(f"artifacts/brick-001-{name}-review")
        path = folder / "brick-001-rough-merged.ply"
        report = json.loads((folder / "merge-report.json").read_text())
        cloud = load(path)
        # Export order is top followed by plate. Old top/plate crops stay preserved.
        n = report["points"]["top"]
        if len(cloud) != sum(report["points"].values()):
            raise ValueError(f"Count mismatch in {name}")
        raw[name] = cloud[:, :3]
        reports[name] = dict(
            source=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pose=report["pose"] if "pose" in report else "initial",
            raw_count=len(cloud),
        )
        for method, points in [("cleaned-both", cloud[:, :3]), ("plate-only", cloud[n:, :3])]:
            cleaned, stats = clean_points(points)
            methods.setdefault(method, {})[name] = cleaned
            reports[name][method] = stats
            save(args.output / f"{method}-{name}.ply", cleaned)
    results = {}
    for method, clouds in methods.items():
        reference = clouds["rotated"]
        aligned = [reference]
        fits = []
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(*reference.T * 1000, s=5, label="upright reference")
        for name in names[1:]:
            moved, metrics = register(clouds[name], reference)
            # Repeated underside is evaluated but omitted from the primary composite.
            if name != "other-side":
                aligned.append(moved)
                ax.scatter(*moved.T * 1000, s=5, label=name)
            metrics["pose"] = name
            fits.append(metrics)
            save(args.output / f"{method}-aligned-{name}.ply", moved)
        merged = np.vstack(aligned)
        save(args.output / f"{method}-composite.ply", merged)
        ax.set(
            xlabel="X (mm)",
            ylabel="Y (mm)",
            zlabel="Z (mm)",
            title=f"{method}: exploratory multi-pose fit",
        )
        ax.set_box_aspect(np.maximum(np.ptp(merged, axis=0), 0.001))
        ax.legend(fontsize=8)
        fig.savefig(args.output / f"{method}-preview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        results[method] = dict(points=len(merged), fits=fits, accepted=False)
    result = dict(
        object_id="brick-001",
        synthetic=False,
        accepted=False,
        python_version=platform.python_version(),
        package_versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "scikit-learn", "matplotlib"]
        },
        inputs=reports,
        methods=results,
        limitations=[
            "Largest-cluster cleanup may discard real disconnected surfaces",
            "Plate-only avoids using approximate cross-camera calibration but retains pose ambiguity",
            "Random point holdout is not independent object or pose validation",
            "Similar repeated faces are excluded from added coverage claims",
            "No invented mesh, missing surfaces remain missing",
        ],
    )
    (args.output / "audit-report.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "review.html").write_text(
        Path(__file__).with_name("reconstruction_review.html").read_text()
    )
    for method, value in results.items():
        print(method, "points", value["points"])
        for fit in value["fits"]:
            print(
                fit["pose"],
                "holdout symmetric mean mm",
                round(fit["symmetric_mean_mm"], 2),
                "best seeds",
                np.round(fit["best_seed_scores_mm"], 2),
            )


if __name__ == "__main__":
    main()
