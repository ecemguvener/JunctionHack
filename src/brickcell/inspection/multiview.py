"""Object-scoped burst decisions; raw scores from different cameras are not averaged."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from brickcell.inspection.types import Decision


@dataclass(frozen=True)
class BurstResult:
    camera: str
    object_id: str
    decision: Decision
    median_score: float | None
    spread: float | None
    threshold: float | None
    count: int
    reason: str


def classify_burst(
    camera: str,
    object_id: str,
    scores: list[float],
    threshold: float | None,
    margin: float = 0,
    min_frames: int = 7,
    calibrated: bool = False,
) -> BurstResult:
    """Require an entire burst's central 80% to agree outside the uncertainty band.

    Burst rules themselves must be validated on real held-out sequences. This
    function never establishes object presence or cross-camera correspondence.
    """
    if not camera or not object_id or min_frames < 3:
        raise ValueError("Camera/object identity and at least three frames are required")

    def result(decision, reason, median=None, spread=None):
        return BurstResult(
            camera, object_id, decision, median, spread, threshold, len(scores), reason
        )

    if not calibrated or threshold is None:
        return result(Decision.REINSPECT, "Camera has no real-image calibration")
    if not math.isfinite(threshold) or threshold <= 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError("Positive finite threshold and nonnegative finite margin required")
    if len(scores) < min_frames or not all(math.isfinite(s) and s >= 0 for s in scores):
        return result(Decision.REINSPECT, "Insufficient valid frames")
    ordered = sorted(scores)
    # Reject isolated outliers while preserving sustained uncertainty/movement.
    trim = len(ordered) // 10
    core = ordered[trim : len(ordered) - trim] if trim else ordered
    median = statistics.median(ordered)
    spread = max(core) - min(core)
    if min(core) > threshold + margin:
        return result(Decision.DEFECTIVE, "Consistent defect across burst", median, spread)
    if max(core) < threshold - margin:
        return result(Decision.GOOD, "Consistent good appearance across burst", median, spread)
    return result(Decision.REINSPECT, "Burst unstable or near threshold", median, spread)


def fuse_views(object_id: str, required: tuple[str, ...], results: list[BurstResult]) -> dict:
    if not object_id or not required or len(set(required)) != len(required):
        raise ValueError("Object identity and unique required cameras are required")
    if any(r.object_id != object_id or r.camera not in required for r in results):
        raise ValueError("Wrong object or unexpected camera in inspection batch")
    if len({r.camera for r in results}) != len(results):
        raise ValueError("Duplicate camera results in inspection batch")
    if any(r.decision == Decision.ERROR for r in results):
        decision, reason = Decision.ERROR, "A camera inspection failed"
    elif any(r.decision == Decision.DEFECTIVE for r in results):
        decision, reason = Decision.DEFECTIVE, "At least one camera consistently detected a defect"
    elif len(results) == len(required) and all(r.decision == Decision.GOOD for r in results):
        decision, reason = Decision.GOOD, "Every required camera consistently accepted this object"
    else:
        decision, reason = Decision.REINSPECT, "Missing, uncalibrated or uncertain camera result"
    return {"object_id": object_id, "decision": str(decision), "reason": reason}
