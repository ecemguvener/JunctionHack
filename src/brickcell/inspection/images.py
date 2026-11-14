from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()
    if min(rgb.shape[:2]) < 16:
        raise ValueError("Image dimensions must be at least 16x16")
    return rgb


def input_tensor(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 -> fixed-size NCHW float32 [0,1]; model owns normalization."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected HWC RGB uint8 image")
    h, w = size
    resized = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


@dataclass(frozen=True)
class QualityConfig:
    min_std: float = 8.0
    min_laplacian_variance: float = 15.0
    background_distance: float = 25.0
    min_foreground_fraction: float = 0.03
    max_foreground_fraction: float = 0.8
    roi: tuple[float, float, float, float] = (0.08, 0.08, 0.92, 0.92)
    max_outside_fraction: float = 0.05


@dataclass(frozen=True)
class ViewQuality:
    valid: bool
    reasons: tuple[str, ...]
    intensity_std: float
    laplacian_variance: float
    foreground_fraction: float
    outside_fraction: float


def check_view(rgb: np.ndarray, config: QualityConfig) -> ViewQuality:
    """Heuristic for an isolated object on a uniform background, not object detection."""
    x0, y0, x1, y1 = config.roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("ROI must be normalized [x0,y0,x1,y1] within [0,1]")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    std = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    background = np.median(border.astype(float), axis=0)
    foreground = (
        np.linalg.norm(rgb.astype(float) - background, axis=-1) > config.background_distance
    )
    h, w = gray.shape
    inside = np.zeros_like(foreground)
    inside[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
    fraction = float(foreground.mean())
    outside = float((foreground & ~inside).sum() / max(1, foreground.sum()))
    reasons = []
    if std < config.min_std or fraction < config.min_foreground_fraction:
        reasons.append("empty_or_low_contrast")
    if blur < config.min_laplacian_variance:
        reasons.append("severe_blur_or_low_texture")
    if fraction > config.max_foreground_fraction or outside > config.max_outside_fraction:
        reasons.append("object_outside_expected_region_or_background_mismatch")
    return ViewQuality(not reasons, tuple(reasons), std, blur, fraction, outside)
