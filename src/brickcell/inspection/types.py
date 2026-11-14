from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    GOOD = "GOOD"
    DEFECTIVE = "DEFECTIVE"
    REINSPECT = "REINSPECT"
    ERROR = "ERROR"


@dataclass(frozen=True)
class InspectionResult:
    object_id: str
    frame_id: str
    timestamp: str
    view_id: str
    is_valid_view: bool
    anomaly_score: float | None
    decision: Decision
    model_version: str
    threshold_version: str | None
    inference_latency_ms: float
    score_type: str = "raw_patchcore_distance_not_probability"
    anomaly_map_path: str | None = None
    reason: str = ""
    inspector_mode: str = "anomalib_openvino"

    def __post_init__(self) -> None:
        if not all(
            (self.object_id, self.frame_id, self.timestamp, self.view_id, self.model_version)
        ):
            raise ValueError("Inspection identity and model version must be nonempty")
        if not isinstance(self.decision, Decision):
            raise ValueError("Inspection decision must be a Decision enum")
        if not math.isfinite(self.inference_latency_ms) or self.inference_latency_ms < 0:
            raise ValueError("Inference latency must be finite and nonnegative")
        if self.anomaly_score is not None and not math.isfinite(self.anomaly_score):
            raise ValueError("Anomaly score must be finite or absent")
