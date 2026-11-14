import numpy as np
import pytest

from brickcell.reconstruction import (
    checked_points,
    clean_points,
    distance_metrics,
    register,
    rigid_fit,
)


def test_rigid_transform_preserves_metric_geometry():
    rng = np.random.default_rng(19)
    points = rng.normal(size=(60, 3)) * 0.02
    angle = 0.7
    expected = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    target = points @ expected + np.array([0.3, -0.2, 0.5])
    rotation, translation = rigid_fit(points, target)
    np.testing.assert_allclose(points @ rotation + translation, target, atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_cleanup_rejects_isolated_stray_points_without_inventing_surfaces():
    rng = np.random.default_rng(7)
    cluster = rng.normal(size=(100, 3)) * 0.001
    points = np.vstack([cluster, [[1, 2, 3], [2, 3, 4]]])
    kept, report = clean_points(points)
    assert report["removed_points"] == 2
    np.testing.assert_allclose(kept, cluster)


def test_bad_clouds_are_not_silently_registered():
    with pytest.raises(ValueError, match="Nonfinite"):
        checked_points(np.full((20, 3), np.nan))
    with pytest.raises(ValueError, match="at least"):
        checked_points(np.zeros((2, 3)))


def test_bidirectional_metric_exposes_partial_surface_collapse():
    x = np.linspace(0, 0.1, 40)
    whole = np.column_stack([x, x * 0, x * 0])
    partial = whole[:15]
    metrics = distance_metrics(partial, whole)
    assert metrics["forward_median_mm"] == 0
    assert metrics["reverse_median_mm"] > 0
    assert metrics["reverse_overlap_fraction"] < metrics["forward_overlap_fraction"]


def test_registration_recovers_known_asymmetric_pose_and_reports_holdout():
    rng = np.random.default_rng(27)
    source = rng.uniform(-1, 1, (150, 3)) * np.array([0.04, 0.015, 0.007])
    angle = 0.4
    expected = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    target = source @ expected + np.array([0.12, 0.25, -0.3])
    moved, report = register(source, target)
    assert report["training_counts"] == [120, 120]
    assert report["held_out_counts"] == [30, 30]
    assert report["validated_pose"] is False
    assert np.linalg.det(report["row_rotation"]) == pytest.approx(1)
    assert np.median(np.linalg.norm(moved - target, axis=1)) < 0.002
