import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brickcell.data.marks import prepare
from brickcell.detector.config import load_config
from brickcell.detector.workflow import verify_sensor_manifest


def dataset(tmp_path):
    data = {
        "cameras": {
            camera: {"serial": camera + "-serial", "crop": [8, 8, 104, 104]}
            for camera in ("plate", "wrist")
        },
        "samples": [],
    }
    rng = np.random.default_rng(123)
    for camera in ("plate", "wrist"):
        for i, (split, label) in enumerate(
            [
                ("train", "good"),
                ("val", "good"),
                ("val", "defective"),
                ("test", "good"),
                ("test", "defective"),
            ]
        ):
            filename = camera + str(i) + ".png"
            Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(
                tmp_path / filename
            )
            data["samples"].append(
                dict(
                    camera=camera,
                    path=filename,
                    object_id=f"brick-{i}",
                    session_id=f"session-{i}",
                    label=label,
                    split=split,
                    view_id="front",
                    brick_type="duplo-2x2",
                    synthetic=True,
                )
            )
    path = tmp_path / "index.json"
    path.write_text(json.dumps(data))
    return path, data


def test_two_camera_preparation_shares_architecture_but_separates_contracts(tmp_path):
    path, _ = dataset(tmp_path)
    result = prepare(path, tmp_path / "prepared")
    assert result["cameras"] == {"plate": 5, "wrist": 5}
    assert result["test_scored"] is False
    for camera in ("plate", "wrist"):
        config = load_config(tmp_path / "prepared" / camera / "patchcore.yaml")
        assert config.image_size == (256, 256)
        assert config.backbone == "resnet18.tv_in1k"
        assert config.minimum_defect_recall == 1.0
        assert config.maximum_good_nonaccept_rate == 0.1
        assert verify_sensor_manifest(config)["serial"] == camera + "-serial"
    plate = load_config(tmp_path / "prepared/plate/patchcore.yaml")
    from dataclasses import replace

    with pytest.raises(ValueError, match="camera/crop"):
        verify_sensor_manifest(
            replace(
                plate,
                sensor_contract=dict(serial="wrist-serial", crop=[8, 8, 104, 104], modality="rgb"),
            )
        )


@pytest.mark.parametrize(
    "problem", ["cross_camera_leak", "duplicate", "defect_train", "missing_wrist", "top"]
)
def test_mark_dataset_rejects_untrusted_partitions_and_unselected_views(tmp_path, problem):
    path, data = dataset(tmp_path)
    if problem == "cross_camera_leak":
        data["samples"][5]["split"] = "val"
    elif problem == "duplicate":
        data["samples"][5]["path"] = data["samples"][0]["path"]
    elif problem == "defect_train":
        data["samples"][0]["label"] = "defective"
    elif problem == "missing_wrist":
        data["samples"] = data["samples"][:5]
    elif problem == "top":
        data["samples"][0]["camera"] = "top"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        prepare(path, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_plate_snapshot_consumes_rgb_without_decoding_depth(tmp_path, monkeypatch):
    import base64
    import io

    module = runpy.run_path("scripts/patchcore_depth_viewer.py")
    config = tmp_path / "viewer.json"
    config.write_text(
        json.dumps(
            {
                "cameras": {
                    "plate": {
                        "type": "rgb",
                        "serial": "plate-serial",
                        "source_format": "snapshot_rgb",
                        "source": "http://example.test/snapshot/plate",
                    }
                }
            }
        )
    )
    image = io.BytesIO()
    Image.new("RGB", (640, 480), "blue").save(image, "JPEG")
    packet = dict(
        synthetic=False,
        serial="plate-serial",
        frame_id="new-1",
        time=time.time(),
        rgb_jpeg_b64=base64.b64encode(image.getvalue()).decode(),
        depth_png_b64="not-depth",
    )
    handlers = []

    class Server:
        def __init__(self, address, handler):
            handlers.append(handler)

        def serve_forever(self):
            pass

    main = module["main"]
    monkeypatch.setitem(main.__globals__, "ThreadingHTTPServer", Server)
    monkeypatch.setitem(
        main.__globals__, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(packet).encode())
    )
    monkeypatch.setattr(sys, "argv", ["viewer", "--config", str(config)])
    main()
    request = object.__new__(handlers[0])
    request.path = "/api/plate"
    responses = []
    request.send = lambda status, body, kind="application/json": responses.append(
        (status, json.loads(body))
    )
    request.do_GET()
    assert responses[-1][0] == 200
    data = responses[-1][1]
    assert set(data["images"]) == {"rgb"}
    assert data["result"]["depth_used_for_anomaly_score"] is False
    assert data["result"]["decision"] == "REINSPECT"


def test_default_mark_inputs_are_plate_and_wrist_rgb_only():
    settings = json.loads(Path("configs/mark-viewer.json").read_text())
    assert set(settings["cameras"]) == {"plate", "wrist"}
    assert all(c["type"] == "rgb" and c["models"] == {} for c in settings["cameras"].values())


def test_preparation_preserves_foreground_training_choice(tmp_path):
    path, data = dataset(tmp_path)
    data["cameras"]["plate"].update(
        surface_profile={"hue_ranges": [[95, 115]]},
        foreground_memory=True,
    )
    path.write_text(json.dumps(data))
    prepare(path, tmp_path / "prepared")
    assert load_config(tmp_path / "prepared/plate/patchcore.yaml").foreground_memory
    assert not load_config(tmp_path / "prepared/wrist/patchcore.yaml").foreground_memory


def test_real_inspector_boundary_excludes_top_and_never_passes_depth():
    """Exercise the actual inspector classes without importing optional robot/Frax SDKs."""
    import ast
    import concurrent.futures
    import math
    from dataclasses import dataclass
    from types import SimpleNamespace

    source = ast.parse(Path("src/brickcell/block_inspector/hardware_cell.py").read_text())
    selected = [
        n
        for n in source.body
        if isinstance(n, ast.ClassDef)
        and n.name in ("FrameInspectionResult", "RealCameraAnomalyInspector")
    ]
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[]
    )
    namespace = {"dataclass": dataclass, "np": np, "math": math, "concurrent": concurrent}
    exec(compile(module, "<actual-inspector-boundary>", "exec"), namespace)
    inspector_type = namespace["RealCameraAnomalyInspector"]
    result_type = namespace["FrameInspectionResult"]
    detector = SimpleNamespace(threshold=SimpleNamespace(value=0.5))
    service = inspector_type(
        camera_detectors={"plate": {"rgb": detector}, "top": {"rgb": detector}}
    )
    calls = []
    service._score_sync = lambda *args: (
        calls.append(args) or result_type(True, 0.1, 0.5, False, [], 100)
    )
    try:
        assert service.configured_cameras() == ["plate"]
        top = service.evaluate_frame(
            np.zeros((32, 32, 3), np.uint8), None, [0, 0, 32, 32], camera_name="top"
        )
        assert not top.valid_view and not top.flagged
        assert calls == []
        service.evaluate_frame(
            np.zeros((32, 32, 3), np.uint8),
            np.ones((32, 32)),
            [0, 0, 32, 32],
            expected_z_m=0.5,
            camera_name="plate",
        )
        assert calls[0][1] is None and calls[0][3] is None
        assert not service.has_detector_for_camera("wrist")
    finally:
        service.close()
    with pytest.raises(ValueError, match="RGB models only"):
        inspector_type(camera_detectors={"wrist": {"depth": detector}})
