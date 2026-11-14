"""Provisional depth-plane registration from reviewed board outline pixels.

Specific to orange-corner-review-02. Not an automatic or accepted calibration.
"""

import json
from pathlib import Path

import cv2
import numpy as np


def plane_corners(packet, corners):
    points = np.asarray(packet["points"], dtype=float)[:, :3]
    i = packet["intrinsics"]
    pixels = np.column_stack(
        (
            points[:, 0] / points[:, 2] * i["fx"] + i["ppx"],
            points[:, 1] / points[:, 2] * i["fy"] + i["ppy"],
        )
    )
    contour = cv2.convexHull(corners.astype(np.float32))
    inside = np.array([cv2.pointPolygonTest(contour, tuple(p), False) >= 0 for p in pixels])
    points = points[inside]
    if len(points) < 50:
        raise ValueError("Insufficient board depth")
    rng = np.random.default_rng(42)
    best = np.zeros(len(points), bool)
    for _ in range(300):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        keep = abs((points - a) @ n) < 0.003
        if keep.sum() > best.sum():
            best = keep
    q = points[best]
    center = q.mean(0)
    _, _, vt = np.linalg.svd(q - center, full_matrices=False)
    n = vt[-1]
    offset = n @ center
    rays = np.column_stack(
        ((corners[:, 0] - i["ppx"]) / i["fx"], (corners[:, 1] - i["ppy"]) / i["fy"], np.ones(4))
    )
    xyz = rays * (offset / (rays @ n))[:, None]
    return xyz, dict(
        roi_points=len(points),
        plane_inliers=len(q),
        plane_rms_mm=float(np.sqrt(np.mean(((q - center) @ n) ** 2)) * 1000),
    )


def main():
    folder = Path("artifacts/orange-corner-review-02")
    top = np.array([[145, 197], [258, 152], [367, 300], [236, 359]], float)
    plate = np.roll(np.array([[157, 327], [365, 245], [581, 274], [525, 407]], float), 1, axis=0)
    packets = {n: json.loads((folder / f"{n}-points.json").read_text()) for n in ["top", "plate"]}
    a, qa = plane_corners(packets["top"], top)
    b, qb = plane_corners(packets["plate"], plate)
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((b - bc).T @ (a - ac))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    R = vt.T @ correction @ u.T
    t = ac - R @ bc
    errors = np.linalg.norm(b @ R.T + t - a, axis=1) * 1000
    report = dict(
        accepted=False,
        synthetic=False,
        method="manual outline + depth plane, provisional",
        top_from_plate_R=R.tolist(),
        top_from_plate_t_m=t.tolist(),
        corner_residual_mm=errors.tolist(),
        top=qa,
        plate=qb,
        limitations=[
            "Four manually selected outline corners; no independent validation",
            "Pinhole ray approximation; lens distortion not corrected",
            "Planar fit and pattern correlation do not establish defect-scale accuracy",
            "Do not use for robot control or metric defect decisions",
        ],
    )
    (folder / "provisional-transform.json").write_text(json.dumps(report, indent=2) + "\n")
    # Export a review cloud in top-camera SDK coordinates, with source colors.
    clouds = []
    for name in ["top", "plate"]:
        cloud = np.asarray(packets[name]["points"], float)
        if name == "plate":
            cloud[:, :3] = cloud[:, :3] @ R.T + t
        clouds.append(cloud)
    cloud = np.vstack(clouds)
    with (folder / "provisional-merged-scene.ply").open("w") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(cloud)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        np.savetxt(f, cloud, fmt=["%.5f"] * 3 + ["%d"] * 3)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
