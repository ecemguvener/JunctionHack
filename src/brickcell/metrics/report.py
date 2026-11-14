from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from brickcell.util import json_default


def latency(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)) if values else None,
        "p50_ms": float(np.percentile(values, 50)) if len(values) >= 2 else None,
        "p95_ms": float(np.percentile(values, 95)) if len(values) >= 20 else None,
        "p95_note": "Requires >=20 samples; includes no hardware claims",
    }


def classification(labels: list[int], scores: list[float], threshold: float) -> dict:
    if not labels or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Evaluation requires matching nonempty labels and finite scores")
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel().tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    both = len(set(labels)) == 2
    return {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": float(precision),
        "recall": float(recall) if tp + fn else None,
        "f1": float(f1),
        "defect_recall": tp / (tp + fn) if tp + fn else None,
        "defect_miss_rate": fn / (tp + fn) if tp + fn else None,
        "good_false_rejection_rate": fp / (fp + tn) if fp + tn else None,
        "auroc": float(roc_auc_score(labels, scores)) if both else None,
        "aupr": float(average_precision_score(labels, scores)) if both else None,
    }


class EventLog:
    def __init__(self, path: Path, verbose: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.verbose = verbose

    def emit(self, event: dict) -> None:
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, default=json_default, allow_nan=False) + "\n")
        if self.verbose:
            print(
                f"[{event.get('object_id', '-')}] {event.get('event', '')}: {event.get('state', event.get('detail', ''))}"
            )
