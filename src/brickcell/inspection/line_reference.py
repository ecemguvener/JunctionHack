"""Localize dark marks against a fixed clean reference; not calibrated PatchCore."""

import cv2
import numpy as np


def dark_residual(reference, image, reference_mask, image_mask, max_shift=3):
    if reference.shape != image.shape or reference.dtype != np.uint8 or image.dtype != np.uint8:
        raise ValueError("Require matched RGB uint8 crops")
    if reference_mask.shape != reference.shape[:2] or image_mask.shape != reference_mask.shape:
        raise ValueError("Object masks must match RGB crops")
    centers = []
    for mask in [reference_mask, image_mask]:
        y, x = np.where(mask)
        if len(x) < 20:
            raise ValueError("Insufficient object mask")
        centers.append(np.array([x.mean(), y.mean()]))
    shift = centers[1] - centers[0]
    if np.linalg.norm(shift) > max_shift:
        raise ValueError("Object moved too far for local reference comparison")
    transform = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
    aligned = cv2.warpAffine(
        image,
        transform,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    support = cv2.erode(reference_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)[:, :, 0].astype(float)
    b = cv2.cvtColor(aligned, cv2.COLOR_RGB2LAB)[:, :, 0].astype(float)
    residual = a - b
    residual -= np.median(residual[support])
    return residual, support, aligned, shift


def localize(reference, image, reference_mask, image_mask, threshold=12, min_area=6):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Positive reference threshold required")
    residual, support, aligned, shift = dark_residual(reference, image, reference_mask, image_mask)
    binary = ((residual > threshold) & support).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = []
    for label in range(1, n):
        x, y, w, h, area = stats[label]
        if area >= min_area:
            boxes.append(
                dict(
                    box=[int(x), int(y), int(x + w), int(y + h)],
                    pixels=int(area),
                    peak_darkening=float(residual[labels == label].max()),
                )
            )
    return (
        dict(
            status="DARK_MARK_CANDIDATE" if boxes else "NO_SUPPORTED_DARK_MARK",
            boxes=boxes,
            threshold_lab_l=float(threshold),
            alignment_shift_pixels=shift.tolist(),
            method="clean RGB reference + translation + brightness-compensated dark residual",
            calibrated=False,
        ),
        residual,
        aligned,
    )
