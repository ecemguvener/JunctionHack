"""Explicit colored-block foreground contract shared by training and inference.

Color identifies the exterior silhouette only. Enclosed dark/gray marks remain
inside the filled silhouette; thresholding color must never erase the defect.
This assumes an isolated colored block in a reviewed crop, not arbitrary scenes.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from brickcell.inspection.images import input_tensor


class SurfaceViewError(ValueError):
    """Cannot establish a usable block surface; request a new view, not a defect sort."""


@dataclass(frozen=True)
class SurfaceProfile:
    hue_ranges: list[list[int]]
    version: str = "colored_block_v1"
    min_saturation: int = 45
    min_value: int = 25
    min_area_fraction: float = 0.12
    max_area_fraction: float = 0.9
    max_secondary_ratio: float = 0.15
    min_object_extent: int = 24
    brightness_target: int = 160
    min_gain: float = 0.5
    max_gain: float = 2.0
    min_laplacian_variance: float = 5.0
    max_clipped_fraction: float = 0.2
    max_saturation: int = 235

    def __post_init__(self):
        if self.version not in {"colored_block_v1", "pale_block_v1"}:
            raise ValueError("Unsupported block-surface profile version")
        if (self.version == "colored_block_v1" and not self.hue_ranges) or any(
            len(pair) != 2
            or any(type(v) is not int for v in pair)
            or not 0 <= pair[0] <= pair[1] <= 179
            for pair in self.hue_ranges
        ):
            raise ValueError("Use inclusive OpenCV hue ranges in [0,179]; split wraparound red")
        if self.version == "pale_block_v1" and self.hue_ranges:
            raise ValueError("Pale-block profiles use saturation separation, not hue ranges")
        if type(self.max_saturation) is not int or not 0 < self.max_saturation < 255:
            raise ValueError("Invalid maximum saturation")
        if not 0 < self.min_area_fraction < self.max_area_fraction < 1:
            raise ValueError("Invalid object area limits")
        if not 0 <= self.max_secondary_ratio < 1 or self.min_object_extent < 16:
            raise ValueError("Invalid object ambiguity/size limits")
        if not all(
            0 < value < 255
            for value in (self.min_saturation, self.min_value, self.brightness_target)
        ):
            raise ValueError("Invalid surface saturation/value parameters")
        if not 0 < self.min_gain <= 1 <= self.max_gain:
            raise ValueError("Invalid illumination gain limits")
        if not np.isfinite(self.min_laplacian_variance) or self.min_laplacian_variance < 0:
            raise ValueError("Invalid surface sharpness limit")
        if not 0 <= self.max_clipped_fraction <= 1:
            raise ValueError("Invalid surface clipping limit")


def isolate_surface(rgb: np.ndarray, profile: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return masked, globally exposure-normalized RGB and a filled object mask.

    No local smoothing, inpainting or contrast equalization is applied to RGB:
    marks and molded surface texture are retained for PatchCore to compare.
    """
    cfg = SurfaceProfile(**profile)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected HWC RGB uint8")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if cfg.version == "pale_block_v1":
        # The reviewed crop must contain one pale block against a saturated
        # backdrop. This also retains the cyan cast from the current blue plate.
        seed = hsv[..., 1] <= cfg.max_saturation
    else:
        seed = np.zeros(rgb.shape[:2], dtype=bool)
        for low, high in cfg.hue_ranges:
            seed |= (hsv[..., 0] >= low) & (hsv[..., 0] <= high)
        seed &= hsv[..., 1] >= cfg.min_saturation
    seed &= hsv[..., 2] >= cfg.min_value
    contours, _ = cv2.findContours(
        seed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours or cv2.contourArea(contours[0]) < 1:
        raise SurfaceViewError("No supported colored block in the crop")
    largest = cv2.contourArea(contours[0])
    if len(contours) > 1 and cv2.contourArea(contours[1]) > cfg.max_secondary_ratio * largest:
        raise SurfaceViewError("Multiple matching regions: block identity/occlusion is ambiguous")
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours[:1], -1, 1, thickness=cv2.FILLED)
    mask = mask.astype(bool)
    x, y, w, h = cv2.boundingRect(contours[0])
    if x == 0 or y == 0 or x + w == rgb.shape[1] or y + h == rgb.shape[0]:
        raise SurfaceViewError("Block touches crop border; surface may be clipped")
    if min(w, h) < cfg.min_object_extent:
        raise SurfaceViewError("Block too small in source pixels for this surface profile")
    if not cfg.min_area_fraction <= mask.mean() <= cfg.max_area_fraction:
        raise SurfaceViewError("Block area outside calibrated crop expectations")
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F)[interior].var())
    if not np.isfinite(sharpness) or sharpness < cfg.min_laplacian_variance:
        raise SurfaceViewError("Block surface is blurred or lacks sufficient visible texture")
    if float((hsv[..., 2][mask] >= 250).mean()) > cfg.max_clipped_fraction:
        raise SurfaceViewError("Too much clipped glare on the block surface")
    # Estimate exposure from colored plastic, excluding enclosed dark/gray marks.
    level = float(np.median(hsv[..., 2][seed & mask]))
    gain = cfg.brightness_target / max(level, 1.0)
    if not cfg.min_gain <= gain <= cfg.max_gain:
        raise SurfaceViewError("Block exposure outside supported normalization range")
    result = np.full_like(rgb, 127)
    result[mask] = np.clip(np.rint(rgb[mask].astype(np.float32) * gain), 0, 255).astype(np.uint8)
    return result, mask


def surface_input(rgb, size, profile=None):
    if profile is None:
        return input_tensor(rgb, size), None
    isolated, mask = isolate_surface(rgb, profile)
    h, w = size
    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not mask.any():
        raise SurfaceViewError("No surface survives model input resizing")
    return input_tensor(isolated, size), mask


def surface_score(score, anomaly_map, mask):
    """Use raw PatchCore map distances only inside the block, with fresh calibration."""
    if mask is None:
        return float(score), anomaly_map
    if anomaly_map is None or anomaly_map.shape != mask.shape or not np.isfinite(anomaly_map).all():
        raise ValueError("Surface scoring requires a finite full-resolution PatchCore anomaly map")
    return float(anomaly_map[mask].max()), np.where(mask, anomaly_map, 0).astype(np.float32)
