"""Color- and plate-aware anomaly block detection for the Intel demo cell.

Detects the three target anomaly DUPLO blocks (white 2x4, lime/yellow-green 2x2,
and orange-brown/lavender stacked 2x2) when outside the blue, green, and pink
KALAS plates or while presented in the gripper during arm rotation.
"""

from __future__ import annotations

import cv2
import numpy as np


def _hsv_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    return {
        "white_block": (s <= 52) & (v >= 175),
        "lime_block": (h >= 26) & (h <= 48) & (s >= 55) & (v >= 110),
        "orange_brown_block": (h >= 6) & (h <= 20) & (s >= 120) & (v >= 70),
        "lavender_block": (h >= 125) & (h <= 152) & (s >= 35) & (s <= 125) & (v >= 110),
    }


def _classify_single_rgb(mean_rgb: tuple[float, float, float]) -> str | None:
    arr = np.uint8([[list(np.clip(np.round(mean_rgb), 0, 255))]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if s <= 55 and v >= 170:
        return "white_block"
    if 26 <= h <= 48 and s >= 55 and v >= 110:
        return "lime_block"
    if 6 <= h <= 21 and s >= 110 and v >= 70:
        return "orange_brown_block"
    if 122 <= h <= 152 and 30 <= s <= 125 and v >= 110:
        return "lavender_block"
    return None


def _plate_interior_mask(rgb: np.ndarray) -> np.ndarray:
    """Build a mask covering the blue (source), green (good), and pink (defective) plates."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    plate_raw = (
        ((h >= 85) & (h <= 118) & (s >= 65) & (v >= 40))  # blue / aqua
        | ((h >= 50) & (h <= 84) & (s >= 55) & (v >= 40))  # green
        | (((h >= 145) | (h <= 5)) & (s >= 45) & (v >= 75))  # pink
    ).astype(np.uint8)
    plate_raw = cv2.morphologyEx(plate_raw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(plate_raw)
    contours, _ = cv2.findContours(plate_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    min_plate_area = max(900, int(0.004 * rgb.shape[0] * rgb.shape[1]))
    if num_labels > 1 and np.any(stats[1:, 4] >= min_plate_area):
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_plate_area:
                hull = cv2.convexHull(cnt)
                cv2.drawContours(mask, [hull], -1, 1, thickness=-1)
    if np.any(mask):
        mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)
    return mask > 0


def classify_held_anomaly_block(
    rgb_crop: np.ndarray,
    mean_rgb: tuple[float, float, float] | None = None,
) -> tuple[bool, str | None, dict[str, float]]:
    """Classify whether a held brick crop (outside all plates) is one of the 3 anomaly blocks."""
    if mean_rgb is not None:
        rgb_label = _classify_single_rgb(mean_rgb)
        if rgb_label is not None:
            return True, rgb_label, {rgb_label: 1.0}

    if rgb_crop.size == 0 or rgb_crop.ndim != 3 or rgb_crop.shape[2] != 3:
        return False, None, {}

    h_img, w_img = rgb_crop.shape[:2]
    y0, y1 = int(0.2 * h_img), max(int(0.2 * h_img) + 1, int(0.8 * h_img))
    x0, x1 = int(0.2 * w_img), max(int(0.2 * w_img) + 1, int(0.8 * w_img))
    center = rgb_crop[y0:y1, x0:x1]
    total = float(max(1, center.shape[0] * center.shape[1]))
    masks = _hsv_masks(center)
    ratios = {name: float(np.sum(mask)) / total for name, mask in masks.items()}

    # Require a meaningful fraction in the tight held-brick center to ignore rim/background.
    thresholds = {
        "white_block": 0.22,
        "lime_block": 0.18,
        "orange_brown_block": 0.18,
        "lavender_block": 0.24,
    }
    best_name: str | None = None
    best_ratio = 0.0
    for name, thr in thresholds.items():
        if ratios[name] >= thr and ratios[name] > best_ratio:
            best_name = name
            best_ratio = ratios[name]
    return best_name is not None, best_name, ratios


def detect_unplated_anomaly_blocks(rgb: np.ndarray) -> tuple[bool, str | None, list[list[int]]]:
    """Detect any of the 3 anomaly blocks in `rgb` when outside the blue/green/pink plates."""
    if rgb.size == 0 or rgb.ndim != 3 or rgb.shape[2] != 3:
        return False, None, []

    h_img, w_img = rgb.shape[:2]
    in_plate = _plate_interior_mask(rgb)
    masks = _hsv_masks(rgb)
    min_area = max(250, int(0.001 * h_img * w_img))
    max_area = int(0.06 * h_img * w_img)

    detected_boxes: list[list[int]] = []
    detected_labels: list[str] = []
    for name in ("white_block", "lime_block", "orange_brown_block"):
        comp_mask = (masks[name] & (~in_plate)).astype(np.uint8)
        comp_mask = cv2.morphologyEx(comp_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        comp_mask = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n_cc, _, stats, _ = cv2.connectedComponentsWithStats(comp_mask)
        for idx in range(1, n_cc):
            x, y, bw, bh, area = map(int, stats[idx])
            if area < min_area or area > max_area:
                continue
            if min(bw, bh) < 16 or max(bw, bh) > int(0.45 * max(w_img, h_img)):
                continue
            # Exclude border-touching desk/wall slabs; held/tabletop anomaly blocks are compact.
            if x <= 2 or y <= 2 or x + bw >= w_img - 2 or y + bh >= h_img - 2:
                continue
            detected_boxes.append([x, y, x + bw, y + bh])
            detected_labels.append(name)

    if not detected_boxes:
        return False, None, []
    return True, detected_labels[0], detected_boxes
