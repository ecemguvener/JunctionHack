"""Attempt trimmed ICP across sparse brick poses for provisional review only.

Symmetry and partial surfaces make ICP ambiguous. Fit residual is not accuracy.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def load_cloud(path):
    with path.open() as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Missing PLY header")
            if line.strip() == "end_header":
                break
        return np.loadtxt(stream)[:, :3]


def fit(source, target):
    tree = cKDTree(target)
    sm, tm = source.mean(0), target.mean(0)
    _, _, sa = np.linalg.svd(source - sm, full_matrices=False)
    _, _, ta = np.linalg.svd(target - tm, full_matrices=False)
    candidates = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            seed = sa.T @ (np.eye(3)[list(perm)] * np.array(signs)[:, None]) @ ta
            if np.linalg.det(seed) < 0:
                continue
            r = seed
            t = tm - sm @ r
            for _ in range(30):
                moved = source @ r + t
                distances, indices = tree.query(moved)
                keep = distances <= np.percentile(distances, 80)
                a, b = source[keep], target[indices[keep]]
                ac, bc = a.mean(0), b.mean(0)
                u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
                fix = np.eye(3)
                fix[-1, -1] = np.linalg.det(u @ vt)
                r = u @ fix @ vt
                t = bc - ac @ r
            moved = source @ r + t
            distances, _ = tree.query(moved)
            score = float(np.mean(np.sort(distances)[: max(3, int(len(distances) * 0.8))]))
            candidates.append((score, moved, r, t))
    candidates.sort(key=lambda a: a[0])
    return candidates[0], [float(c[0] * 1000) for c in candidates[:3]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clouds", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clouds = [load_cloud(p) for p in args.clouds]
    target = clouds[0]
    aligned = [target]
    report = []
    for path, cloud in zip(args.clouds[1:], clouds[1:]):
        best, scores = fit(cloud, target)
        score, moved, r, t = best
        aligned.append(moved)
        report.append(
            dict(
                source=str(path),
                trimmed_fit_distance_mm=score * 1000,
                best_three_seed_scores_mm=scores,
                row_rotation=r.tolist(),
                translation_m=t.tolist(),
            )
        )
    merged = np.vstack(aligned)
    with (args.output / "brick-001-candidate-composite.ply").open("w") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(merged)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        )
        np.savetxt(stream, merged, fmt="%.6f")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    for path, cloud in zip(args.clouds, aligned):
        ax.scatter(*cloud.T * 1000, s=5, label=path.parent.name.replace("brick-001-", ""))
    ax.set(
        xlabel="X (mm)",
        ylabel="Y (mm)",
        zlabel="Z (mm)",
        title="Candidate multi-pose alignment — unverified ICP",
    )
    ax.set_box_aspect(np.maximum(np.ptp(merged, axis=0), 0.001))
    ax.legend(fontsize=7)
    fig.savefig(args.output / "candidate-preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    result = dict(
        accepted=False,
        object_id="brick-001",
        synthetic=False,
        source_clouds=[str(p) for p in args.clouds],
        fits=report,
        limitations=[
            "Symmetric partial surfaces can align incorrectly",
            "Trimmed fit is measured on fitting points; not independent accuracy",
            "No complete mesh or verified hidden-face coverage",
        ],
    )
    (args.output / "alignment-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
