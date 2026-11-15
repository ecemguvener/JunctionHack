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


def test_calibrated_patchcore_event_bridges_to_camui():
    bad = bridge.detection_payload(
        event(
            decision="DEFECTIVE",
            model_version="6d8add3e0688a925",
            threshold_version="4128199aec09eedb",
            raw_distance=2.61,
            threshold=2.4538,
        ),
        now=101,
    )
    assert bad["malformed"] is True
    assert bad["source"] == "patchcore_openvino_6d8add3e0688a925"
    assert "DEFECT FLAGGED" in bad["detail"]
    good = bridge.detection_payload(
        event(
            decision="GOOD",
            model_version="6d8add3e0688a925",
            threshold_version="4128199aec09eedb",
            raw_distance=1.82,
            threshold=2.4538,
        ),
        now=101,
    )
    assert good["malformed"] is False
    assert good["source"] == "patchcore_openvino_6d8add3e0688a925"
    assert "GOOD" in good["detail"]


def test_camui_reporter_posts_patchcore_source_and_frame_to_detection_store(monkeypatch):
    from brickcell.block_inspector.camera_adapters import CamUiReporter

    store = ui.Detection(5.0)
    reporter = CamUiReporter("http://127.0.0.1:8093")

    def fake_post(body: dict) -> bool:
        store.update(body)
        return True

    monkeypatch.setattr(reporter, "_post_payload", fake_post)
    ok = reporter.report(
        malformed=False,
        detail="GOOD | PatchCore 6d8add3e0688a925 raw_dist=1.820 vs thr 2.454",
        camera="side",
        frame_id="243622060450-1001",
        source="patchcore_openvino_6d8add3e0688a925",
    )
    assert ok is True
    status = store.status()
    assert status["state"] == "ok"
    assert status["camera"] == "side"
    assert status["frame_id"] == "243622060450-1001"
    assert status["source"] == "patchcore_openvino_6d8add3e0688a925"
    assert "PatchCore 6d8add3e0688a925" in status["detail"]


def test_camui_tracks_side_and_wrist_independently():
    store = ui.Detection(5.0)
    store.update(
        {
            "malformed": True,
            "detail": "DEFECT FLAGGED | PatchCore 2.610",
            "camera": "side",
            "frame_id": "243622060450-1",
            "source": "patchcore_openvino_6d8add3e0688a925",
        }
    )
    store.update(
        {
            "malformed": False,
            "detail": "GOOD | Wrist PatchCore 1.920",
            "camera": "wrist",
            "frame_id": "sonix-wrist-1",
            "source": "patchcore_openvino_6d8add3e0688a925",
        }
    )
    status = store.status()
    assert status["state"] == "malformed"
    assert status["camera"] == "side"
    assert status["cameras"]["side"]["state"] == "malformed"
    assert status["cameras"]["wrist"]["state"] == "ok"


def test_wrist_event_bridges_to_camui():
    payload = bridge.detection_payload(
        event(
            camera="wrist",
            serial="sonix-wrist",
            decision="DEFECTIVE",
            model_version="6d8add3e0688a925",
            threshold_version="4128199aec09eedb",
            raw_distance=2.71,
            threshold=2.4538,
        ),
        now=101,
    )
    assert payload["camera"] == "wrist"
    assert payload["malformed"] is True
    assert payload["source"] == "patchcore_openvino_6d8add3e0688a925"
    assert "Wrist PatchCore" in payload["detail"]


def test_wrist_anomaly_babyros_node_integrates_with_camui(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from brickcell.block_inspector.camera_adapters import CamUiReporter
    from brickcell.block_inspector.runtime_bus import (
        CANONICAL_TOPICS,
        InProcessRuntimeBus,
        pack_rgbd_frame_nddc,
    )
    from brickcell.block_inspector.runtime_nodes import WristAnomalyNode

    store = ui.Detection(5.0)
    reporter = CamUiReporter("http://127.0.0.1:8093")
    monkeypatch.setattr(reporter, "_post_payload", lambda body: (store.update(body), True)[1])
    reporter.report(
        malformed=False,
        detail="GOOD | Plate PatchCore 2.100",
        camera="side",
        frame_id="243622060450-10",
        source="patchcore_openvino_6d8add3e0688a925",
    )

    scores = iter([2.71, 2.12])
    detector = SimpleNamespace(
        model_version="6d8add3e0688a925",
        metadata={"synthetic": False},
        threshold=SimpleNamespace(
            value=2.453849196434021,
            version="4128199aec09eedb",
            decide=lambda sc: "DEFECTIVE" if sc >= 2.453849196434021 else "GOOD",
        ),
        predict=lambda roi: SimpleNamespace(score=next(scores), latency_ms=18.5, anomaly_map=None),
    )
    bus = InProcessRuntimeBus()
    published = []
    bus.subscribe(CANONICAL_TOPICS["wrist_anomaly"], published.append)
    node = WristAnomalyNode(bus=bus, detector=detector, camui=reporter)

    rgb = np.full((64, 64, 3), 180, dtype=np.uint8)
    bus.publish(CANONICAL_TOPICS["wrist_frame"], pack_rgbd_frame_nddc(rgb, 1_000))
    status1 = store.status()
    assert len(published) == 1
    assert published[0]["decision"] == "DEFECTIVE"
    assert status1["state"] == "malformed"
    assert status1["cameras"]["side"]["state"] == "ok"
    assert status1["cameras"]["wrist"]["state"] == "malformed"

    bus.publish(CANONICAL_TOPICS["wrist_frame"], pack_rgbd_frame_nddc(rgb, 2_000))
    status2 = store.status()
    assert published[1]["decision"] == "GOOD"
    assert status2["state"] == "ok"
    assert status2["cameras"]["wrist"]["state"] == "ok"
    assert status2["cameras"]["side"]["state"] == "ok"
    node.close()
    bus.close()
