import dataclasses
from datetime import UTC, datetime

import pytest

from brickcell.inspection.types import Decision, InspectionResult
from brickcell.integrations.case_builder import build
from brickcell.integrations.core import evidence


def test_existing_inspector_result_becomes_hashed_case_without_reclassification(
    image_case, tmp_path
):
    source = InspectionResult(
        object_id="brick-1",
        frame_id="frame-1",
        timestamp=datetime.fromtimestamp(100, UTC).isoformat(),
        view_id="front",
        is_valid_view=False,
        anomaly_score=None,
        decision=Decision.REINSPECT,
        model_version="model-1",
        threshold_version=None,
        inference_latency_ms=10,
    )
    case = build(
        dataclasses.asdict(source),
        "session-1",
        "wrist",
        100,
        False,
        {"observed": tmp_path / "observed.png"},
        tmp_path,
    )
    assert case["decision"] == "REINSPECT"
    assert case["frame_id"] == source.frame_id
    assert case["score"] is None
    with pytest.raises(ValueError, match="acquisition"):
        build(dataclasses.asdict(source), "session-1", "wrist", 101, False, {}, tmp_path)
    (tmp_path / "observed.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        evidence(case["assets"][0], tmp_path)
