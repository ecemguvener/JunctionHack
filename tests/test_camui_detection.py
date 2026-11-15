import importlib.util

import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ui = module("camera_ui", "camera-ui/camui.py")
bridge = module("camui_bridge", "scripts/publish_camui_detection.py")


def event(**changes):
    return (
        dict(
            camera="plate",
            serial="243622060450",
            synthetic=False,
            frame_id="real-frame-1",
            captured_at=100,
            decision="UNCALIBRATED",
            model_version="dark-mark-heuristic-real",
            raw_distance=20,
            contrast=16,
        )
        | changes
    )


def test_mark_warning_and_clear_have_real_source_and_frame():
    found = bridge.detection_payload(event(), now=101)
    assert found["malformed"] is True
    assert found["captured_at"] == 100
    assert found["frame_id"] == "real-frame-1"
    assert found["camera"] == "side"
    assert "not_patchcore" in found["source"]
    assert bridge.detection_payload(event(raw_distance=10), now=101)["malformed"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"synthetic": True},
        {"camera": "top"},
        {"serial": "other"},
        {"frame_id": None},
        {"captured_at": 95},
        {"captured_at": 102},
        {"captured_at": float("nan")},
        {"decision": "REINSPECT"},
        {"raw_distance": None},
        {"contrast": 0},
        {"model_version": "unknown"},
    ],
)
def test_untrusted_missing_stale_or_wrong_sensor_never_turns_red(changes):
    assert bridge.detection_payload(event(**changes), now=101)["malformed"] is None


def test_warning_expires_from_capture_not_repeat_post(monkeypatch):
    monkeypatch.setattr(ui.time, "time", lambda: 101)
    store = ui.Detection(5)
    payload = bridge.detection_payload(event(), now=101)
    store.update(payload)
    assert store.status()["state"] == "malformed"
    monkeypatch.setattr(ui.time, "time", lambda: 104)
    store.update(payload)
    monkeypatch.setattr(ui.time, "time", lambda: 106)
    assert store.status()["state"] == "stale"
    with pytest.raises(ValueError, match="stale"):
        store.update(payload)
    store.update(bridge.detection_payload(event(), now=106))
    assert store.status()["state"] == "none"


@pytest.mark.parametrize("value", [1, 0, "true", []])
def test_warning_requires_real_boolean(value):
    with pytest.raises(ValueError):
        ui.Detection(5).update({"malformed": value})
