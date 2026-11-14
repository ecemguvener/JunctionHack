"""Conservative evidence gate for the deployed Plate RGB-D inspection path.

Depth support is necessary here, but is not proof of brick identity, a successful
grasp, or complete surface coverage. RGB-only previews need a separate presence
validator before they can be used to sort an object.
"""

from dataclasses import dataclass
from numbers import Integral

import cv2
import numpy as np


@dataclass(frozen=True)
class ViewEvidence:
    valid_view: bool
    reason: str
    foreground_pixels: int = 0
    foreground_mask: np.ndarray | None = None

    @property
    def depth_supported(self) -> bool:
        return self.valid_view and self.foreground_mask is not None


def validate_plate_view(
    rgb: np.ndarray,
    depth_m: np.ndarray | None,
    crop_box: list[int],
    expected_z_m: float | None,
    z_band_m: float,
    *,
    min_foreground_pixels: int = 45,
    min_foreground_fraction: float = 0.10,
    min_interior_pixels: int = 50,
) -> ViewEvidence:
    """Validate aligned metric depth and an inspectable crop before scoring.

    The colour exclusions and 7px erosion match the existing dark-mark scorer.
    A blue/purple object may consequently be rejected; it must not silently get a
    GOOD decision because that scorer inspected no surface. Defaults are initial
    conservative settings, not thresholds calibrated on real captures.
    """
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
    ):
        return ViewEvidence(False, "invalid_rgb")
    if (
        not isinstance(crop_box, (list, tuple))
        or len(crop_box) != 4
        or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in crop_box)
    ):
        return ViewEvidence(False, "invalid_crop")
    x0, y0, x1, y1 = map(int, crop_box)
    h, w = rgb.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        return ViewEvidence(False, "crop_out_of_bounds")
    roi = rgb[y0:y1, x0:x1]
    if min(roi.shape[:2]) < 20 or float(roi.std()) <= 4.0:
        return ViewEvidence(False, "empty_or_low_contrast_roi")
    if depth_m is None:
        return ViewEvidence(False, "missing_depth")
    if (
        not isinstance(depth_m, np.ndarray)
        or depth_m.ndim != 2
        or depth_m.shape != rgb.shape[:2]
        or not np.issubdtype(depth_m.dtype, np.floating)
    ):
        return ViewEvidence(False, "invalid_aligned_metric_depth")
    try:
        expected = float(expected_z_m)
        band = float(z_band_m)
    except (TypeError, ValueError):
        return ViewEvidence(False, "invalid_expected_depth")
    if not np.isfinite(expected) or expected <= 0:
        return ViewEvidence(False, "invalid_expected_depth")
    if not np.isfinite(band) or not 0 < band < expected:
        return ViewEvidence(False, "invalid_depth_band")
    if min_foreground_pixels < 1 or min_interior_pixels < 1 or not 0 < min_foreground_fraction <= 1:
        raise ValueError("Foreground evidence requirements must be positive and fraction <= 1")

    roi_depth = depth_m[y0:y1, x0:x1]
    mask = np.isfinite(roi_depth) & (roi_depth > 0) & (np.abs(roi_depth - expected) <= band)
    count = int(mask.sum())
    if count < min_foreground_pixels or count / mask.size < min_foreground_fraction:
        return ViewEvidence(False, "insufficient_depth_support", count)

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    plate = (hsv[:, :, 0] >= 92) & (hsv[:, :, 0] <= 114) & (hsv[:, :, 1] >= 95)
    jaw = (hsv[:, :, 0] >= 118) & (hsv[:, :, 0] <= 148) & (hsv[:, :, 1] >= 75)
    interior = cv2.erode((mask & ~plate & ~jaw).astype(np.uint8), np.ones((7, 7), np.uint8))
    if int(interior.sum()) < min_interior_pixels:
        return ViewEvidence(False, "no_inspectable_surface", count)
    return ViewEvidence(True, "depth_supported_roi", count, mask)
