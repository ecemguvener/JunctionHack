import io
import json
import runpy

import pytest


@pytest.mark.parametrize(
    "offset,frame_id,accepted",
    [(0, "frame", True), (-10, "frame", False), (10, "frame", False), (0, "", False)],
)
def test_capture_fetch_refuses_stale_future_or_missing_identity(
    monkeypatch, offset, frame_id, accepted
):
    fetch = runpy.run_path("scripts/capture_rgbd_sequence.py")["fetch"]
    monkeypatch.setattr(fetch.__globals__["time"], "time", lambda: 100)
    monkeypatch.setitem(
        fetch.__globals__,
        "urlopen",
        lambda *_args, **_kw: io.StringIO(json.dumps({"time": 100 + offset, "frame_id": frame_id})),
    )
    if accepted:
        assert fetch("http://127.0.0.1:18094", "plate")["frame_id"] == "frame"
    else:
        with pytest.raises(ValueError):
            fetch("http://127.0.0.1:18094", "plate")
