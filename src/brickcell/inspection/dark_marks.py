"""Image-only dark-stroke heuristic; no robot commands or calibrated defect claims."""

from typing import Any

import cv2
import numpy as np


def score_brick_roi_dark_mark(
    roi_rgb: np.ndarray,
    fg_mask: np.ndarray | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Evaluate symmetric-flanked CIE-Lab blackhat raw surface anomaly score on a brick ROI.

    Detects thin dark pen/marker strokes or surface gouges flanked on both transverse sides
    by uniform brick plastic (same lightness and Lab chromaticity), rejecting outer silhouette
    edges, plate rims, circular stud shadows, and underside hollow tubes.
    Never returns or represents a probability: `raw_score` is a Lab contrast distance (>= 0).
    """
    if roi_rgb.size == 0 or min(roi_rgb.shape[:2]) < 20:
        return 0.0, []

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    # Exclude aquamarine/blue KALAS plate plastic and purple PLA robot gripper jaws
    not_plate = ~((hsv[:, :, 0] >= 92) & (hsv[:, :, 0] <= 114) & (hsv[:, :, 1] >= 95))
    not_purple_jaw = ~((hsv[:, :, 0] >= 118) & (hsv[:, :, 0] <= 148) & (hsv[:, :, 1] >= 75))
    valid_fg = (
        not_plate & not_purple_jaw if fg_mask is None else (fg_mask & not_plate & not_purple_jaw)
    )

    interior = cv2.erode(valid_fg.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if int(np.sum(interior)) < 50:
        return 0.0, []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, kernel) * interior

    cand = (blackhat >= 14.0) & interior
    n_comp, lbls, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8))
    best_score = 0.0
    detections: list[dict[str, Any]] = []
    h_img, w_img = L.shape

    for idx in range(1, n_comp):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 9 or area > 280:
            continue
        ys, xs = np.where(lbls == idx)
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        centered = pts - pts.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(1, len(pts) - 1)
        evals, evecs = np.linalg.eigh(cov)
        l1 = float(max(evals[-1], 1e-6))
        l2 = float(max(evals[0], 0.0))
        linearity = (l1 - l2) / l1
        _, (w_r, h_r), _ = cv2.minAreaRect(pts.astype(np.float32))
        maj = max(float(w_r), float(h_r))
        min_s = max(min(float(w_r), float(h_r)), 1.0)
        if maj < 7.5 or min_s > 6.8 or linearity < 0.70:
            continue

        normal = evecs[:, 0]
        p_plus = np.round(pts + 5.0 * normal[None, :]).astype(int)
        p_minus = np.round(pts - 5.0 * normal[None, :]).astype(int)
        in_bounds = (
            (p_plus[:, 0] >= 0)
            & (p_plus[:, 0] < w_img)
            & (p_plus[:, 1] >= 0)
            & (p_plus[:, 1] < h_img)
            & (p_minus[:, 0] >= 0)
            & (p_minus[:, 0] < w_img)
            & (p_minus[:, 1] >= 0)
            & (p_minus[:, 1] < h_img)
        )
        if int(np.sum(in_bounds)) < 6:
            continue

        pp = p_plus[in_bounds]
        pm = p_minus[in_bounds]
        pv = pts[in_bounds].astype(int)
        both_fg = valid_fg[pp[:, 1], pp[:, 0]] & valid_fg[pm[:, 1], pm[:, 0]]
        if float(np.mean(both_fg)) < 0.72:
            continue

        l_plus = L[pp[:, 1], pp[:, 0]]
        l_minus = L[pm[:, 1], pm[:, 0]]
        l_mid = L[pv[:, 1], pv[:, 0]]
        side_l_diff = float(np.median(np.abs(l_plus - l_minus)))
        ab_diff = float(
            np.median(
                np.hypot(
                    A[pp[:, 1], pp[:, 0]] - A[pm[:, 1], pm[:, 0]],
                    B[pp[:, 1], pp[:, 0]] - B[pm[:, 1], pm[:, 0]],
                )
            )
        )
        flank_contrast = float(np.median(np.minimum(l_plus, l_minus) - l_mid))
        if side_l_diff > 14.5 or ab_diff > 12.5 or flank_contrast < 12.0:
            continue

        raw_score = float(flank_contrast * (0.5 + 0.5 * linearity))
        x0 = int(stats[idx, cv2.CC_STAT_LEFT])
        y0 = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        detections.append(
            {
                "box": [x0, y0, x0 + bw, y0 + bh],
                "score": round(raw_score, 4),
                "label": "dark mark",
                "linearity": round(linearity, 3),
            }
        )
        best_score = max(best_score, raw_score)

    return best_score, detections
