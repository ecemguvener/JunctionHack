import base64
import copy
import json
import runpy
from pathlib import Path

import pytest

from brickcell.integration.evidence import provenance_errors

AUDIT = runpy.run_path(str(Path(__file__).parents[1] / "scripts/audit_station.py"))


def bundle():
    def response(data):
        return dict(ok=True, received_at=100.5, data=data)

    return dict(
        schema_version=1,
        samples=[
            {
                "top": response(dict(frame_id="t1", serial="top-serial", time=100)),
                "plate": response(dict(frame_id="p1", serial="plate-serial", time=100)),
                "dashboard": response(
                    dict(
                        anomalies=dict(
                            connected=True,
                            source="inspector",
                            age_s=1,
                            items=[dict(box=[10, 10, 30, 30], score=16.85)],
                        ),
                        focus=dict(connected=True, object_id="part-001"),
                    )
                ),
            }
        ],
    )


def codes(data):
    return {f["code"] for f in AUDIT["analyze"](data)["findings"]}


def test_connected_dashboard_is_not_proof_of_fresh_inference():
    result = AUDIT["analyze"](bundle())
    assert "OVERLAY_UNBOUND_TO_CAPTURE" in codes(bundle())
    assert "DETECTOR_PROVENANCE_MISSING" in codes(bundle())
    assert not result["detection_accuracy_evaluated"]
    assert not result["physical_success_verified"]


def test_repeated_frame_and_refreshed_old_overlay_are_visible():
    data = bundle()
    data["samples"].append(copy.deepcopy(data["samples"][0]))
    data["samples"][1]["dashboard"]["data"]["anomalies"]["age_s"] = 0.1
    assert {"REPEATED_CAMERA_FRAME", "POSSIBLE_HEARTBEAT_REFRESH"} <= codes(data)


def test_identical_results_with_increasing_age_do_not_claim_heartbeat():
    data = bundle()
    data["samples"].append(copy.deepcopy(data["samples"][0]))
    data["samples"][1]["dashboard"]["data"]["anomalies"]["age_s"] = 2
    assert "POSSIBLE_HEARTBEAT_REFRESH" not in codes(data)


def test_down_services_and_expired_results_are_reported():
    data = bundle()
    data["samples"][0]["top"] = dict(ok=False, error="timeout")
    data["samples"][0]["dashboard"]["data"]["focus"]["connected"] = False
    assert {"SOURCE_UNAVAILABLE", "OVERLAY_NOT_CURRENT"} <= codes(data)


@pytest.mark.parametrize("stamp", [0, 105, None])
def test_capture_age_is_not_receive_time(stamp):
    data = bundle()
    data["samples"][0]["plate"]["data"]["time"] = stamp
    assert codes(data) & {"CAMERA_AGE_OR_CLOCK_SKEW", "CAPTURE_TIME_MISSING"}


def test_run_success_remains_unverified():
    data = bundle()
    data["run_report"] = dict(data=dict(succeeded=True, sorted_counts=dict(good=1, defective=6)))
    assert "RUN_SUCCESS_IS_SELF_REPORTED" in codes(data)


def test_render_escapes_source_strings_and_never_draws_boxes():
    data = bundle()
    data["samples"][0]["top"]["data"].update(
        frame_id="<script>alert(1)</script>",
        rgb_jpeg_b64=base64.b64encode(b"\xff\xd8fixture").decode(),
    )
    page = AUDIT["render"](data, AUDIT["analyze"](data))
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "<canvas" not in page and "<svg" not in page
    assert "not proven to be the detector input" in page


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://host",
        "http://localhost/x",
        "http://user:secret@localhost:8096",
        "http://localhost/?x=1",
    ],
)
def test_audit_is_limited_to_local_service_urls(url):
    with pytest.raises(ValueError, match="loopback"):
        AUDIT["base_url"](url)


def test_collect_reads_only_known_endpoints(monkeypatch):
    seen = []

    def get(url, timeout):
        seen.append(url)
        return dict(ok=True, data={}, received_at=100)

    monkeypatch.setitem(AUDIT["collect"].__globals__, "get_json", get)
    result = AUDIT["collect"](samples=1)
    assert len(result["samples"]) == 1
    assert {url.split("/", 3)[-1] for url in seen} == {
        "api/status",
        "snapshot/top",
        "snapshot/plate",
        "health",
    }
    assert len(seen) == 5


def test_offline_replay_writes_portable_report_and_preserves_existing(tmp_path):
    source = tmp_path / "capture.json"
    source.write_text(json.dumps(bundle()))
    output = tmp_path / "review"
    assert AUDIT["main"](["replay", "--bundle", str(source), "--output", str(output)]) == 0
    assert (output / "index.html").exists()
    assert json.loads((output / "bundle.json").read_text()) == bundle()
    with pytest.raises(SystemExit):
        AUDIT["main"](["replay", "--bundle", str(source), "--output", str(output)])


def evidence():
    frame = dict(object_id="brick-1", frame_id="capture-1", camera_id="plate", captured_at=100)
    result = dict(
        **frame,
        coordinate_space="source_image_pixels",
        method="anomalib_patchcore",
        model_version="model-1",
        threshold_version="val-1",
        decision="DEFECTIVE",
    )
    return result, frame


def test_proposed_provenance_contract_accepts_matching_capture():
    result, frame = evidence()
    assert provenance_errors(result, frame, 101) == []


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("camera_id", "top", "mismatched_camera_id"),
        ("object_id", "brick-2", "mismatched_object_id"),
        ("frame_id", "old", "mismatched_frame_id"),
        ("captured_at", 50, "stale_or_future_acquisition"),
        ("coordinate_space", "projected_object_box", "unverified_coordinate_space"),
        ("threshold_version", None, "uncalibrated_decision"),
    ],
)
def test_provenance_contract_rejects_bad_binding(key, value, expected):
    result, frame = evidence()
    result[key] = value
    assert expected in provenance_errors(result, frame, 101)


def test_heartbeat_time_does_not_replace_acquisition_time():
    result, frame = evidence()
    result["received_at"] = 120
    assert "stale_or_future_acquisition" in provenance_errors(result, frame, 120)
