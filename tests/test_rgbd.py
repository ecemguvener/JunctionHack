import numpy as np
import pytest

from brickcell.detector.runtime import Prediction
from brickcell.inspection.images import QualityConfig
from brickcell.inspection.rgbd import (
    DEPTH_ENCODING,
    ReferenceRGBInspector,
    RGBDInspector,
    encode_depth,
)
from brickcell.inspection.threshold import Threshold


def frame():
    rgb = np.zeros((64, 64, 3), np.uint8) + 30
    rgb[16:48, 16:48] = [30, 150, 240]
    depth = np.full((64, 64), 500, np.uint16)
    return rgb, depth


class Detector:
    model_version = "test-real"

    def __init__(self, modality, score=0.1, calibrated=True):
        contract = dict(serial="cam", crop=[0, 0, 64, 64], modality=modality)
        if modality == "depth":
            contract["depth_encoding"] = DEPTH_ENCODING
        self.metadata = dict(synthetic=False, sensor_contract=contract)
        self.threshold = Threshold(0.5, 0.02, "threshold", "hash", 10) if calibrated else None
        self.score = score
        self.inputs = []

    def predict(self, image):
        self.inputs.append(image.copy())
        return Prediction(self.score, 1, np.zeros((16, 16)))


def reference_detector():
    detector = Detector("rgb", calibrated=False)
    detector.metadata = dict(
        synthetic=False,
        camera="plate",
        scope="SAME_OBJECT_REFERENCE_PILOT_NOT_ACCURACY_EVALUATION",
        automatic_sorting_allowed=False,
    )
    return detector


def test_reference_review_scores_variable_crops_without_sorting_permission():
    detector = reference_detector()
    for crop in ([0, 0, 64, 64], [4, 4, 60, 60]):
        service = ReferenceRGBInspector(
            "cam",
            crop,
            {"rgb": detector},
            quality=QualityConfig(min_laplacian_variance=0, max_outside_fraction=1),
        )
        result, previews = service.inspect(frame()[0], 100, "cam", "frame", now=101)
        assert result["decision"] == "UNCALIBRATED"
        assert result["automatic_sorting_allowed"] is False
        assert result["channels"]["rgb"]["threshold_version"] is None
        assert result["channels"]["rgb"]["score"] == detector.score
        assert "rgb_map" in previews


@pytest.mark.parametrize("problem", ["stale", "wrong_camera", "no_crop", "invalid_crop"])
def test_reference_review_rejects_untrusted_views_before_inference(problem):
    detector = reference_detector()
    crop = None if problem == "no_crop" else [0, 0, 64, 64]
    if problem == "invalid_crop":
        crop = [0, 0, 1000, 1000]
    service = ReferenceRGBInspector("cam", crop, {"rgb": detector})
    result, _ = service.inspect(
        frame()[0],
        100,
        "wrong" if problem == "wrong_camera" else "cam",
        "frame",
        now=110 if problem == "stale" else 101,
    )
    assert result["decision"] in ("ERROR", "REINSPECT")
    assert result["automatic_sorting_allowed"] is False
    assert not detector.inputs


@pytest.mark.parametrize("problem", ["threshold", "synthetic", "wrist", "unknown", "permission"])
def test_reference_mode_requires_explicit_real_uncalibrated_plate_model(problem):
    detector = reference_detector()
    if problem == "threshold":
        detector.threshold = Threshold(0.5, 0, "test", "manifest", 2)
    elif problem == "synthetic":
        detector.metadata["synthetic"] = True
    elif problem == "wrist":
        detector.metadata["camera"] = "wrist"
    elif problem == "unknown":
        detector.metadata.pop("scope")
    elif problem == "permission":
        detector.metadata["automatic_sorting_allowed"] = True
    with pytest.raises(ValueError, match="uncalibrated real plate"):
        ReferenceRGBInspector("cam", detectors={"rgb": detector})


def inspector(detectors=None):
    return RGBDInspector(
        "cam",
        [0, 0, 64, 64],
        detectors,
        quality=QualityConfig(min_laplacian_variance=0, max_outside_fraction=1),
    )


def inspect(service, depth=None, **kw):
    rgb, d = frame()
    return service.inspect(
        rgb, d if depth is None else depth, 0.001, 100, "cam", "frame-1", now=kw.get("now", 101)
    )


def test_depth_encoding_fixed_range_and_missing_values():
    image, valid = encode_depth(np.array([[0, 150, 500, 1500, 1600]], np.uint16), 0.001)
    assert valid.tolist() == [[False, True, True, True, False]]
    assert image[0, 0].tolist() == [0, 0, 0]
    assert image[0, 1].tolist() == [1, 1, 1]
    assert image[0, 3].tolist() == [255, 255, 255]
    np.testing.assert_array_equal(image[:, :, 0], image[:, :, 1])


def test_rgb_and_depth_use_different_inputs_and_fuse_only_calibrated_results():
    rgb, depth = Detector("rgb"), Detector("depth", 0.8)
    result, previews = inspect(inspector(dict(rgb=rgb, depth=depth)))
    assert result["decision"] == "DEFECTIVE"
    assert result["depth_used_for_anomaly_score"] is True
    assert not np.array_equal(rgb.inputs[0], depth.inputs[0])
    assert "depth_map" in previews


def test_no_models_or_uncalibrated_depth_cannot_produce_good():
    result, _ = inspect(inspector())
    assert result["decision"] == "REINSPECT"
    assert result["depth_used_for_anomaly_score"] is False
    result, _ = inspect(
        inspector(dict(rgb=Detector("rgb"), depth=Detector("depth", calibrated=False)))
    )
    assert result["decision"] == "REINSPECT"


def test_invalid_depth_and_stale_frames_never_call_models():
    rgb = Detector("rgb")
    service = inspector(dict(rgb=rgb))
    result, _ = inspect(service, np.zeros((64, 64), np.uint16))
    assert result["decision"] == "REINSPECT"
    result, _ = inspect(service, now=105)
    assert result["decision"] == "ERROR"
    assert not rgb.inputs


def test_synthetic_and_wrong_modality_contract_rejected():
    model = Detector("rgb")
    model.metadata["synthetic"] = True
    with pytest.raises(ValueError):
        inspector(dict(rgb=model))
    with pytest.raises(ValueError):
        inspector(dict(depth=Detector("rgb")))


def test_model_error_and_nonfinite_map_block_result():
    rgb = Detector("rgb")
    rgb.score = float("nan")
    result, previews = inspect(inspector(dict(rgb=rgb)))
    assert result["decision"] == "ERROR"
    assert result["channels"] == {} and not previews


def test_missing_crop_is_explicit_reinspect():
    result, _ = inspect(RGBDInspector("cam"))
    assert result["decision"] == "REINSPECT"
    assert "crop" in result["reason"]


def test_valid_both_models_can_accept_but_wrong_camera_cannot():
    service = inspector(dict(rgb=Detector("rgb"), depth=Detector("depth")))
    result, _ = inspect(service)
    assert result["decision"] == "GOOD"
    rgb, depth = frame()
    result, _ = service.inspect(rgb, depth, 0.001, 100, "different-camera", "frame", now=101)
    assert result["decision"] == "ERROR"


def test_snapshot_decoder_preserves_metric_depth_and_rejects_wrong_alignment():
    import base64
    import io
    import runpy

    from PIL import Image

    decode = runpy.run_path("scripts/patchcore_depth_viewer.py")["decode_snapshot"]
    rgb, depth = frame()
    depth[0, 0] = 1234
    a, b = io.BytesIO(), io.BytesIO()
    Image.fromarray(rgb).save(a, format="JPEG")
    Image.fromarray(depth).save(b, format="PNG")
    packet = dict(
        synthetic=False,
        aligned_to="color",
        rgb_jpeg_b64=base64.b64encode(a.getvalue()).decode(),
        depth_png_b64=base64.b64encode(b.getvalue()).decode(),
    )
    _, decoded = decode(packet)
    np.testing.assert_array_equal(decoded, depth)
    packet["aligned_to"] = "depth"
    with pytest.raises(ValueError, match="aligned"):
        decode(packet)


@pytest.mark.parametrize(
    ("score", "calibrated", "expected"),
    [(0.1, True, "GOOD"), (0.8, True, "DEFECTIVE"), (0.1, False, "REINSPECT")],
)
def test_arm_rgb_inspection_without_depth(score, calibrated, expected):
    from brickcell.inspection.rgbd import RGBInspector

    detector = Detector("rgb", score, calibrated)
    service = RGBInspector(
        "cam",
        [0, 0, 64, 64],
        {"rgb": detector},
        quality=QualityConfig(min_laplacian_variance=0, max_outside_fraction=1),
    )
    rgb, _ = frame()
    result, previews = service.inspect(rgb, 100, "cam", "wrist-1", now=101)
    assert result["decision"] == expected
    assert set(result["channels"]) == {"rgb"}
    assert result["depth_used_for_anomaly_score"] is False
    assert set(previews) == {"rgb", "rgb_map"}
    np.testing.assert_array_equal(detector.inputs[0], rgb)


def test_arm_rejects_wrong_camera_stale_frames_and_depth_models():
    from brickcell.inspection.rgbd import RGBInspector

    detector = Detector("rgb")
    service = RGBInspector("cam", [0, 0, 64, 64], {"rgb": detector})
    rgb, _ = frame()
    for timestamp, serial, frame_id in [(95, "cam", "id"), (100, "other", "id"), (100, "cam", "")]:
        result, _ = service.inspect(rgb, timestamp, serial, frame_id, now=101)
        assert result["decision"] == "ERROR"
    assert not detector.inputs
    with pytest.raises(ValueError):
        RGBInspector("cam", [0, 0, 64, 64], {"depth": Detector("depth")})


def test_arm_missing_model_or_crop_and_out_of_bounds_crop():
    from brickcell.inspection.rgbd import RGBInspector

    rgb, _ = frame()
    for crop in [None, [0, 0, 64, 64], [0, 0, 1920, 1080]]:
        service = RGBInspector(
            "cam",
            crop,
            quality=QualityConfig(min_laplacian_variance=0, max_outside_fraction=1),
        )
        result, _ = service.inspect(rgb, 100, "cam", "id", now=101)
        assert result["decision"] == ("ERROR" if crop and crop[2] == 1920 else "REINSPECT")
