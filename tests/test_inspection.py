from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from brickcell.inspection.images import QualityConfig, check_view, input_tensor, read_rgb
from brickcell.inspection.service import InspectionService
from brickcell.inspection.threshold import Threshold, calibrate
from brickcell.inspection.types import Decision, InspectionResult
from brickcell.sources.directory import Frame


def test_rgb_no_bgr_or_double_normalization(tmp_path):
    path = tmp_path / "red.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(path)
    value = input_tensor(read_rgb(path), (64, 64))
    assert value.shape == (1, 3, 64, 64)
    assert value.dtype == np.float32
    assert value[0, 0, 0, 0] == 1
    assert value[0, 2, 0, 0] == 0


def test_gray_and_rgba(tmp_path):
    for mode in ("L", "RGBA"):
        path = tmp_path / f"{mode}.png"
        Image.new(mode, (32, 32)).save(path)
        assert read_rgb(path).shape == (32, 32, 3)
    with pytest.raises(ValueError):
        input_tensor(np.zeros((32, 32)), (64, 64))
    with pytest.raises(FileNotFoundError):
        read_rgb(tmp_path / "missing.png")


def test_empty_blurred_and_outside_roi(manifest):
    assert not check_view(np.zeros((128, 128, 3), dtype="uint8"), QualityConfig()).valid
    rgb = read_rgb(manifest.image_path(manifest.samples[0]))
    assert check_view(rgb, QualityConfig()).valid
    blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(12)))
    assert not check_view(blurred, QualityConfig()).valid
    outside = np.roll(rgb, 35, axis=1)
    assert not check_view(outside, QualityConfig()).valid


def test_threshold_and_uncertainty():
    threshold = Threshold(2, 0.1, "v1", "hash", 10)
    assert threshold.decide(1) == Decision.GOOD
    assert threshold.decide(3) == Decision.DEFECTIVE
    assert threshold.decide(2) == Decision.REINSPECT
    assert threshold.decide(float("nan")) == Decision.ERROR
    assert Threshold(2, 0, "v1", "h", 10).decide(2) == Decision.DEFECTIVE
    with pytest.raises(ValueError):
        Threshold(float("nan"), 0, "x", "x", 1)


def test_calibration_requires_both_classes():
    with pytest.raises(ValueError):
        calibrate([1, 2], [0, 0], "h")
    with pytest.raises(ValueError):
        calibrate([1, 1], [0, 1], "h")
    threshold = calibrate([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], "h", 0.02)
    assert 0.2 < threshold.value < 0.8
    assert threshold.uncertainty_margin > 0


def test_result_structure():
    result = InspectionResult("o", "f", "time", "front", True, 2.0, Decision.GOOD, "m", "t", 3.2)
    assert asdict(result)["score_type"] == "raw_patchcore_distance_not_probability"


def test_no_threshold_is_error(manifest):
    from brickcell.detector.runtime import Prediction

    class Detector:
        model_version = "test"
        threshold = None

        def predict(self, rgb):
            return Prediction(0.1, 1, None)

    service = InspectionService(Detector())
    frame = Frame(
        "o",
        "f",
        datetime.now(timezone.utc).isoformat(),
        "front",
        manifest.image_path(manifest.samples[0]),
    )
    result = service.inspect(frame)
    assert result.decision == Decision.ERROR
    assert "threshold" in result.reason
    bad = service.inspect(Frame("o", "bad", "now", "front", Path("missing-file.png")))
    assert bad.decision == Decision.ERROR


def test_invalid_view_does_not_call_detector(tmp_path):
    class Detector:
        model_version = "test"
        threshold = None

        def predict(self, rgb):
            pytest.fail("Invalid views must not reach anomaly inference")

    path = tmp_path / "empty.png"
    Image.new("RGB", (64, 64)).save(path)
    result = InspectionService(Detector()).inspect(Frame("o", "f", "now", "front", path))
    assert result.decision == Decision.REINSPECT
    assert result.anomaly_score is None


def test_invalid_result_fields():
    with pytest.raises(ValueError, match="finite"):
        InspectionResult("o", "f", "t", "v", True, float("nan"), Decision.GOOD, "m", "t", 1)
    with pytest.raises(ValueError, match="Decision enum"):
        InspectionResult("o", "f", "t", "v", True, 1, "surprise", "m", "t", 1)
