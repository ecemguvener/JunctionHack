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
    scores: list[float],
    labels: list[int],
    manifest_hash: str,
    uncertainty_fraction: float = 0.0,
    *,
    minimum_defect_recall: float | None = None,
    maximum_good_nonaccept_rate: float = 0.1,
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
    method = "validation_max_f1_ties_favor_recall"
    if minimum_defect_recall is not None:
        if not 0 < minimum_defect_recall <= 1 or not 0 <= maximum_good_nonaccept_rate <= 1:
            raise ValueError("Recall must be in (0,1] and good nonaccept rate in [0,1]")
        # Decisions change at score +/- margin. Include boundaries and their
        # neighbors to respect inclusive REINSPECT edges, including tied scores.
        boundaries = np.unique(np.concatenate((unique - margin, unique + margin)))
        candidates = sorted(
            {
                float(t)
                for boundary in boundaries
                for t in (boundary, np.nextafter(boundary, -np.inf), np.nextafter(boundary, np.inf))
                if math.isfinite(t)
            }
        )
        feasible = []
        for candidate in candidates:
            trial = Threshold(candidate, margin, "candidate", manifest_hash, len(scores))
            decisions = [trial.decide(s) for s in scores]
            recall = sum(
                d == Decision.DEFECTIVE for d, label in zip(decisions, labels) if label == 1
            ) / labels.count(1)
            nonaccept = sum(
                d != Decision.GOOD for d, label in zip(decisions, labels) if label == 0
            ) / labels.count(0)
            if recall >= minimum_defect_recall and nonaccept <= maximum_good_nonaccept_rate:
                feasible.append((nonaccept, -recall, candidate))
        if not feasible:
            raise ValueError(
                "Validation cannot meet defect recall and clean nonaccept limits; "
                "collect better views/data or improve the model. No threshold approved."
            )
        threshold = min(feasible)[2]
        method = "validation_recall_constrained"
    payload = dict(
        value=threshold,
        uncertainty_margin=margin,
        calibration_manifest_sha256=manifest_hash,
        sample_count=len(scores),
    )
    if minimum_defect_recall is not None:
        payload["method"] = method
    version = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return Threshold(version=version, **payload)


def threshold_dict(threshold: Threshold) -> dict:
    return asdict(threshold)
