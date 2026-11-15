"""Depth-aware PatchCore review with explicit per-modality input contracts."""

import math
import time

import numpy as np

from brickcell.inspection.images import QualityConfig, check_view
from brickcell.inspection.types import Decision

DEPTH_ENCODING = dict(near_m=0.15, far_m=1.5, invalid_value=0, channels="repeat3")


def encode_depth(depth, scale, encoding=DEPTH_ENCODING):
    if depth.ndim != 2 or depth.dtype != np.uint16 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Expected uint16 aligned depth and positive finite metre scale")
    near, far = encoding["near_m"], encoding["far_m"]
    if (
        not 0 < near < far
        or encoding.get("channels") != "repeat3"
        or encoding.get("invalid_value") != 0
    ):
        raise ValueError("Unsupported depth encoding")
    metres = depth.astype(np.float64) * scale
    valid = (depth > 0) & (metres >= near) & (metres <= far)
    channel = np.where(valid, 1 + np.clip((metres - near) / (far - near), 0, 1) * 254, 0).astype(
        np.uint8
    )
    return np.repeat(channel[:, :, None], 3, axis=2), valid


class RGBDInspector:
    """Viewer-only inspector: separate real RGB/depth detectors, no robot actions."""

    modalities = ("rgb", "depth")

    def __init__(self, serial, crop=None, detectors=None, min_depth_fraction=0.6, quality=None):
        self.serial, self.crop, self.detectors = serial, crop, detectors or {}
        self.quality = quality or QualityConfig()
        self.min_depth_fraction = min_depth_fraction
        if not 0 < min_depth_fraction <= 1:
            raise ValueError("Depth fraction must be in (0,1]")
        for modality, detector in self.detectors.items():
            if modality not in self.modalities or detector.metadata.get("synthetic", True):
                raise ValueError("Only explicitly real RGB/depth models are supported")
            expected = dict(serial=serial, crop=crop, modality=modality)
            if modality == "depth":
                expected["depth_encoding"] = DEPTH_ENCODING
            if detector.metadata.get("sensor_contract") != expected:
                raise ValueError("Model sensor contract does not match camera/crop/modality")

    def inspect(self, rgb, depth, scale, captured_at, serial, frame_id, now=None):
        now = time.time() if now is None else now
        result = dict(
            frame_id=frame_id,
            serial=serial,
            captured_at=captured_at,
            decision="REINSPECT",
            reason="",
            channels={},
            depth_fraction=None,
            score_type="raw_patchcore_distance_not_probability",
            depth_used_for_anomaly_score=False,
        )
        previews = {}
        try:
            if (
                not frame_id
                or serial != self.serial
                or not math.isfinite(captured_at)
                or not -0.5 <= now - captured_at <= 3
            ):
                raise ValueError("Missing frame identity, wrong camera or stale/future frame")
            if (
                rgb.dtype != np.uint8
                or rgb.ndim != 3
                or rgb.shape[2] != 3
                or ("depth" in self.modalities and depth.shape != rgb.shape[:2])
            ):
                raise ValueError("RGB and aligned depth shapes/types do not match")
            if self.crop is None:
                result["reason"] = "Set an explicit brick crop before inference"
                return result, previews
            x0, y0, x1, y1 = self.crop
            h, w = rgb.shape[:2]
            if (
                not all(isinstance(v, int) for v in self.crop)
                or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h)
                or min(x1 - x0, y1 - y0) < 16
            ):
                raise ValueError("Crop must be inside image and at least 16x16")
            rgb = rgb[y0:y1, x0:x1]
            previews["rgb"] = rgb
            inputs = {"rgb": rgb}
            fraction = None
            if "depth" in self.modalities:
                depth = depth[y0:y1, x0:x1]
                encoded, valid = encode_depth(depth, scale)
                inputs["depth"] = encoded
                previews["depth"] = encoded
                fraction = float(valid.mean())
                result["depth_fraction"] = fraction
            quality = check_view(rgb, self.quality)
            if not quality.valid or (fraction is not None and fraction < self.min_depth_fraction):
                result["reason"] = "Invalid RGB view or insufficient depth: " + ",".join(
                    quality.reasons
                )
                return result, previews
            for modality, image in inputs.items():
                detector = self.detectors.get(modality)
                if detector is None:
                    result["channels"][modality] = dict(
                        decision="REINSPECT",
                        score=None,
                        reason=f"No trained real {modality} model configured",
                    )
                    continue
                prediction = detector.predict(image)
                if not math.isfinite(prediction.score):
                    raise ValueError("Nonfinite anomaly score")
                threshold = detector.threshold
                decision = threshold.decide(prediction.score) if threshold else Decision.REINSPECT
                result["channels"][modality] = dict(
                    decision=str(decision),
                    score=prediction.score,
                    model_version=detector.model_version,
                    threshold_version=threshold.version if threshold else None,
                    reason="" if threshold else "No calibrated threshold",
                )
                if modality == "depth":
                    result["depth_used_for_anomaly_score"] = True
                if prediction.anomaly_map is not None:
                    amap = np.asarray(prediction.anomaly_map)
                    if not np.isfinite(amap).all():
                        raise ValueError("Nonfinite anomaly map")
                    previews[modality + "_map"] = amap
            decisions = [c["decision"] for c in result["channels"].values()]
            if len(decisions) == len(self.modalities) and all(
                d in ("GOOD", "DEFECTIVE") for d in decisions
            ):
                result["decision"] = "DEFECTIVE" if "DEFECTIVE" in decisions else "GOOD"
            result["reason"] = (
                "Per-modality decisions; no cross-camera coverage or object identity verified"
            )
        except Exception as exc:
            result.update(decision="ERROR", reason=str(exc))
            result["channels"] = {}
            result["depth_used_for_anomaly_score"] = False
            previews = {}
        return result, previews


class RGBInspector(RGBDInspector):
    """RGB-only camera review with the same crop, provenance and quality gates."""

    modalities = ("rgb",)

    def inspect(self, rgb, captured_at, serial, frame_id, now=None):
        return super().inspect(rgb, None, None, captured_at, serial, frame_id, now)
