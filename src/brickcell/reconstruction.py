"""Offline, exploratory registration of captured point clouds.

These diagnostics do not authorize inspection decisions or robot actions.
"""

import itertools

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def checked_points(points, minimum=12):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"Need at least {minimum} XYZ points")
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite point coordinates")
    return points


def clean_points(points, radius_m=0.009, min_neighbors=3):
    """Keep the largest locally connected cluster, preserving raw coordinates."""
    points = checked_points(points)
    labels = DBSCAN(eps=radius_m, min_samples=min_neighbors).fit_predict(points)
    clusters = [(int((labels == label).sum()), label) for label in set(labels) if label >= 0]
    if not clusters:
        raise ValueError("No supported connected surface")
    _, label = max(clusters)
    kept = points[labels == label]
    return checked_points(kept), dict(
        input_points=len(points),
        retained_points=len(kept),
        removed_points=len(points) - len(kept),
        clusters=len(clusters),
        radius_m=radius_m,
    )


def rigid_fit(source, target):
    """Return row-vector proper rotation and translation for paired XYZ."""
    a, b = checked_points(source, 3), checked_points(target, 3)
    if a.shape != b.shape:
        raise ValueError("Corresponding arrays must have equal shape")
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    return rotation, bc - ac @ rotation


def distance_metrics(source, target, overlap_m=0.008):
    source, target = checked_points(source, 3), checked_points(target, 3)
    forward = cKDTree(target).query(source)[0]
    reverse = cKDTree(source).query(target)[0]
    return dict(
        forward_median_mm=float(np.median(forward) * 1000),
        reverse_median_mm=float(np.median(reverse) * 1000),
        symmetric_mean_mm=float((forward.mean() + reverse.mean()) * 500),
        forward_overlap_fraction=float((forward < overlap_m).mean()),
        reverse_overlap_fraction=float((reverse < overlap_m).mean()),
        overlap_radius_mm=overlap_m * 1000,
    )


def register(source, target, seed=42):
    """Multistart trimmed ICP, with held-out points excluded from fitting.

    Holdout measures point interpolation only, not independent pose accuracy.
    """
    source, target = checked_points(source, 25), checked_points(target, 25)
    rng = np.random.default_rng(seed)
    ai, bi = rng.permutation(len(source)), rng.permutation(len(target))
    train_a, test_a = source[ai[: int(0.8 * len(ai))]], source[ai[int(0.8 * len(ai)) :]]
    train_b, test_b = target[bi[: int(0.8 * len(bi))]], target[bi[int(0.8 * len(bi)) :]]
    _, _, sa = np.linalg.svd(train_a - train_a.mean(0), full_matrices=False)
    _, _, sb = np.linalg.svd(train_b - train_b.mean(0), full_matrices=False)
    tree = cKDTree(train_b)
    candidates = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            rotation = sa.T @ (np.eye(3)[list(perm)] * np.array(signs)[:, None]) @ sb
            if np.linalg.det(rotation) < 0:
                continue
            translation = train_b.mean(0) - train_a.mean(0) @ rotation
            for _ in range(50):
                moved = train_a @ rotation + translation
                distances, indices = tree.query(moved)
                keep = distances <= np.percentile(distances, 85)
                new_r, new_t = rigid_fit(train_a[keep], train_b[indices[keep]])
                delta = np.linalg.norm(new_t - translation) + np.linalg.norm(new_r - rotation)
                rotation, translation = new_r, new_t
                if delta < 1e-8:
                    break
            score = distance_metrics(train_a @ rotation + translation, train_b)["symmetric_mean_mm"]
            candidates.append((score, rotation, translation))
    candidates.sort(key=lambda item: item[0])
    score, rotation, translation = candidates[0]
    metrics = distance_metrics(test_a @ rotation + translation, test_b)
    metrics.update(
        fitting_symmetric_mean_mm=score,
        best_seed_scores_mm=[c[0] for c in candidates[:5]],
        row_rotation=rotation.tolist(),
        translation_m=translation.tolist(),
        seed=seed,
        training_counts=[len(train_a), len(train_b)],
        held_out_counts=[len(test_a), len(test_b)],
        validated_pose=False,
    )
    return source @ rotation + translation, metrics
