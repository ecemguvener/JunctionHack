import io
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize("metadata", [True, False])
def test_wrist_raw_source_preserves_rgb_and_requires_capture_metadata(monkeypatch, metadata):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/patchcore_depth_viewer.py"))
    stream = io.BytesIO()
    Image.fromarray(np.full((1080, 1920, 3), [20, 80, 150], np.uint8)).save(stream, "PNG")

    class Response(io.BytesIO):
        headers = (
            {
                "X-Frame-Time": "100",
                "X-Frame-Id": "wrist-001",
                "X-Camera-Serial": "sonix-wrist",
            }
            if metadata
            else {}
        )

    reader = module["read_rgb_frame"]
    monkeypatch.setitem(
        reader.__globals__, "urlopen", lambda source, timeout: Response(stream.getvalue())
    )
    rgb, packet = reader("http://example.test/frame.jpg")
    assert rgb.shape == (1080, 1920, 3)
    assert rgb[0, 0].tolist() == [20, 80, 150]
    assert packet["frame_id"] == ("wrist-001" if metadata else "")
    assert packet["time"] == (100 if metadata else 0)
    assert packet["serial"] == ("sonix-wrist" if metadata else "")


def test_default_viewer_includes_arm_with_separate_rgb_model_configuration():
    root = Path(__file__).parents[1]
    cameras = json.loads((root / "configs/rgbd-viewer.json").read_text())["cameras"]
    assert set(cameras) == {"top", "plate", "wrist"}
    assert cameras["wrist"]["type"] == "rgb"
    assert cameras["wrist"]["models"] == {}


def test_viewer_routes_arm_preview_and_full_resolution_crop(monkeypatch, tmp_path):
    import sys
    import time
    from email.message import Message

    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/patchcore_depth_viewer.py"))
    config = tmp_path / "viewer.json"
    config.write_text(
        json.dumps(
            {
                "cameras": {
                    "wrist": {
                        "type": "rgb",
                        "serial": "sonix-wrist",
                        "source": "http://example.test/frame.jpg",
                    }
                }
            }
        )
    )
    handlers = []

    class Server:
        def __init__(self, address, handler):
            handlers.append(handler)

        def serve_forever(self):
            pass

    main = module["main"]
    monkeypatch.setitem(main.__globals__, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(sys, "argv", ["viewer", "--config", str(config)])
    main()
    handler = object.__new__(handlers[0])
    responses = []
    handler.send = lambda status, body, kind="application/json": responses.append((status, body))
    handler.path = "/crop/wrist"
    body = json.dumps({"crop": [600, 300, 1200, 900]}).encode()
    handler.headers = Message()
    handler.headers["Content-Type"] = "application/json"
    handler.headers["Content-Length"] = str(len(body))
    handler.rfile = io.BytesIO(body)
    handler.do_POST()
    assert responses[-1][0] == 200
    assert json.loads(responses[-1][1])["crop"] == [600, 300, 1200, 900]
    rgb = np.full((1080, 1920, 3), 100, np.uint8)
    monkeypatch.setitem(
        main.__globals__,
        "read_rgb_frame",
        lambda source: (
            rgb,
            {"time": 0, "frame_id": "", "serial": ""},
        ),
    )
    handler.path = "/api/wrist"
    handler.do_GET()
    assert responses[-1][0] == 200
    payload = json.loads(responses[-1][1])
    assert payload["result"]["decision"] == "REINSPECT"
    assert "capture metadata missing" in payload["result"]["reason"]
    assert set(payload["images"]) == {"rgb"}
    monkeypatch.setitem(
        main.__globals__,
        "read_rgb_frame",
        lambda source: (
            rgb,
            {"time": time.time() - 10, "frame_id": "old-frame", "serial": "sonix-wrist"},
        ),
    )
    handler.do_GET()
    payload = json.loads(responses[-1][1])
    assert payload["result"]["decision"] == "REINSPECT"
    assert payload["result"]["channels"] == {}
    handler.path = "/"
    handler.do_GET()
    assert b"CAMERA_NAMES" not in responses[-1][1]
    assert b'["wrist"]' in responses[-1][1]
