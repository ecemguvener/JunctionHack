from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score

from brickcell.inspection.types import Decision


@dataclass(frozen=True)
class Threshold:
    value: float
    uncertainty_margin: float
    version: str
    calibration_manifest_sha256: str
    sample_count: int
    method: str = "validation_max_f1_ties_favor_recall"
    score_type: str = "raw_patchcore_distance_not_probability"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.value)
            or not math.isfinite(self.uncertainty_margin)
            or self.uncertainty_margin < 0
        ):
            raise ValueError("Threshold and nonnegative uncertainty margin must be finite")

    def decide(self, score: float) -> Decision:
        if not math.isfinite(score):
            return Decision.ERROR
        if self.uncertainty_margin > 0 and abs(score - self.value) <= self.uncertainty_margin:
            return Decision.REINSPECT
        return Decision.DEFECTIVE if score >= self.value else Decision.GOOD


def calibrate(
    scores: list[float], labels: list[int], manifest_hash: str, uncertainty_fraction: float = 0.0
) -> Threshold:
    if len(scores) != len(labels) or set(labels) != {0, 1} or not all(map(math.isfinite, scores)):
        raise ValueError(
            "Calibration requires finite validation scores and both good and defective labels"
        )
    if not 0 <= uncertainty_fraction <= 0.5:
        raise ValueError("uncertainty_fraction must be in [0,.5]")
    unique = np.unique(scores)
    if len(unique) < 2:
        raise ValueError("No score separation: cannot calibrate a meaningful threshold")
    candidates = [float(unique[0]), *map(float, (unique[:-1] + unique[1:]) / 2)]
    threshold = max(candidates, key=lambda t: f1_score(labels, np.asarray(scores) >= t))
    margin = float(uncertainty_fraction * (np.quantile(scores, 0.95) - np.quantile(scores, 0.05)))
    payload = dict(
        value=threshold,
        uncertainty_margin=margin,
        calibration_manifest_sha256=manifest_hash,
        sample_count=len(scores),
    )
    version = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return Threshold(version=version, **payload)


def threshold_dict(threshold: Threshold) -> dict:
    return asdict(threshold)
