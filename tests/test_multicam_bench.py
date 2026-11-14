import runpy
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest


def bench(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/multicam_server.py"))
    instance = module["Bench"].__new__(module["Bench"])
    instance.output = tmp_path
    instance.models = {}
    instance.capture_lock = threading.Lock()

    class Camera:
        crop = [0, 0, 32, 32]

        def snapshot(self):
            image = np.full((32, 32, 3), (10, 20, 30), np.uint8)
            return [(time.time(), image) for _ in range(10)], "", self.crop

    instance.cameras = {name: Camera() for name in module["CAMERAS"]}
    return instance


def test_uncalibrated_real_capture_never_becomes_good_from_human_label(tmp_path):
    instance = bench(tmp_path)
    result = instance.capture(
        {
            "confirmed": True,
            "object_id": "brick-1",
            "session": "session-1",
            "label": "good",
        }
    )
    assert result["decision"] == "REINSPECT"
    assert set(result["captures"]) == {"overhead", "side", "wrist"}
    assert all(view["decision"] == "REINSPECT" for view in result["views"])
    paths = list((tmp_path / "raw").rglob("*.png"))
    assert len(paths) == 3
    assert all(cv2.imread(str(path))[0, 0].tolist() == [10, 20, 30] for path in paths)
    assert len({path.name for path in paths}) == 1  # Same capture batch across views.


def test_capture_requires_confirmed_object_and_explicit_crops(tmp_path):
    instance = bench(tmp_path)
    with pytest.raises(ValueError, match="Confirm"):
        instance.capture({"confirmed": False})
    instance.cameras["side"].crop = None
    with pytest.raises(ValueError, match="crop"):
        instance.capture(
            {
                "confirmed": True,
                "object_id": "brick-1",
                "session": "session-1",
                "label": "good",
            }
        )


def test_single_camera_collection_does_not_claim_three_view_acceptance(tmp_path):
    instance = bench(tmp_path)
    instance.cameras["overhead"].crop = None
    result = instance.capture(
        {
            "confirmed": True,
            "object_id": "brick-1",
            "session": "session-1",
            "label": "good",
            "cameras": ["side"],
        }
    )
    assert list(result["captures"]) == ["side"]
    assert result["decision"] == "REINSPECT"
    assert len(list((tmp_path / "raw").rglob("*.png"))) == 1
