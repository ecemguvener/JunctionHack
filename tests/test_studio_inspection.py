"""Frame pairing, acquisition freshness and uncertainty in the display-only bridge."""

import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from brickcell.studio.inspection import InspectionEvidence
from brickcell.studio.server import handle_studio_get, handle_studio_post


def evidence(**overrides):
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), "yellow").save(stream, format="JPEG")
    image = base64.b64encode(stream.getvalue()).decode()
    return (
        dict(
            camera="plate",
            serial="test-camera",
            frame_id="frame-1",
            captured_at=100,
            synthetic=False,
            model_version="test-hash",
            decision="UNCALIBRATED",
            threshold_version=None,
            reason="Preview only",
            raw_distance=2.1,
            image_jpeg=image,
            overlay_jpeg=image,
        )
        | overrides
    )


def test_frame_pair_expires_by_acquisition_not_receipt():
    store = InspectionEvidence()
    store.publish(evidence(), now=102)
    first = store.snapshot(now=102)["cameras"]["plate"]
    assert first["fresh"] and first["image_jpeg"] and first["overlay_jpeg"]
    stale = store.snapshot(now=104)["cameras"]["plate"]
    assert stale["decision"] == "STALE"
    assert stale["overlay_jpeg"] is None and stale["image_jpeg"] is None
    assert store.snapshot(now=104)["automatic_sorting_allowed"] is False
    with pytest.raises(ValueError, match="repeated"):
        store.publish(evidence(captured_at=102), now=102)


@pytest.mark.parametrize(
    "overrides",
    [
        {"decision": "GOOD"},
        {"decision": "DEFECTIVE"},
        {"synthetic": True},
        {"raw_distance": float("nan")},
        {"camera": "top"},
        {"captured_at": 90},
        {"captured_at": 101},
        {"overlay_jpeg": "not-a-jpeg"},
        {"decision": "REINSPECT"},
    ],
)
def test_invalid_evidence_is_rejected(overrides):
    with pytest.raises(ValueError):
        InspectionEvidence().publish(evidence(**overrides), now=100)


def test_routes_do_not_call_robot_and_isolate_camera_records(monkeypatch):
    monkeypatch.setattr("brickcell.studio.inspection.time.time", lambda: 100)
    engine = SimpleNamespace()  # No robot methods exist on this test engine.
    assert handle_studio_post(engine, "/api/so101/inspection", evidence())[0] == 200
    assert handle_studio_post(engine, "/api/so101/inspection", evidence(camera="wrist"))[0] == 200
    response = handle_studio_get(engine, "/api/so101/inspection")
    body = json.loads(response[2])
    assert set(body["cameras"]) == {"plate", "wrist"}
    assert body["cameras"]["plate"]["decision"] == "UNCALIBRATED"
    assert handle_studio_post(engine, "/api/so101/inspection", evidence())[0] == 400


def test_replayed_older_frame_cannot_be_retimestamped_after_newer_frame():
    store = InspectionEvidence()
    store.publish(evidence(), now=100)
    store.publish(evidence(frame_id="frame-2", captured_at=101), now=101)
    with pytest.raises(ValueError, match="repeated"):
        store.publish(evidence(captured_at=102), now=102)
