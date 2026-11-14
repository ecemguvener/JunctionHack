"""Validate observation provenance without interpreting scores or commanding a robot."""

import math


def provenance_errors(result: dict, frame: dict, now: float, max_age_s: float = 3) -> list[str]:
    """Proposed publisher contract; freshness refers to acquisition, never a heartbeat.

    Passing this check is NOT permission to move: grasp verification, calibrated
    decisions and complete face coverage remain responsibilities of the supervisor.
    Missing legacy fields are rejected, never invented from a dashboard receive time.
    """
    errors = []
    for key in ("object_id", "frame_id", "camera_id"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            errors.append(f"missing_{key}")
        elif result[key] != frame.get(key):
            errors.append(f"mismatched_{key}")
    captured = result.get("captured_at")
    if type(captured) not in (int, float) or not math.isfinite(captured):
        errors.append("missing_acquisition_time")
    else:
        if captured != frame.get("captured_at"):
            errors.append("mismatched_acquisition_time")
        if not -0.5 <= now - captured <= max_age_s:
            errors.append("stale_or_future_acquisition")
    if result.get("coordinate_space") != "source_image_pixels":
        errors.append("unverified_coordinate_space")
    if not result.get("method") or not result.get("model_version"):
        errors.append("missing_method_or_model")
    if result.get("decision") in ("GOOD", "DEFECTIVE") and not result.get("threshold_version"):
        errors.append("uncalibrated_decision")
    return errors
