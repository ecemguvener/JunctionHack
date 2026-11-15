"""Controlled pixel fixtures verify isolation, not real camera defect accuracy."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from brickcell.inspection.block_surface import (
    SurfaceProfile,
    SurfaceViewError,
    isolate_surface,
    surface_input,
    surface_score,
)

PROFILE = {"hue_ranges": [[95, 120]], "min_saturation": 90, "min_laplacian_variance": 0}


def block(background=80):
    rgb = np.full((96, 96, 3), background, dtype=np.uint8)
    rgb[20:76, 22:74] = [20, 90, 180]
    return rgb


@pytest.mark.parametrize("background", [0, 255, (160, 100, 50)])
def test_background_changes_do_not_change_model_pixels(background):
    baseline, mask = surface_input(block(), (128, 128), PROFILE)
    other, other_mask = surface_input(block(background), (128, 128), PROFILE)
    np.testing.assert_array_equal(baseline, other)
    np.testing.assert_array_equal(mask, other_mask)


@pytest.mark.parametrize("mark", [(0, 0, 0), (65, 65, 65)])
def test_color_mask_keeps_dark_and_gray_marks_inside_object(mark):
    rgb = block()
    rgb[40:48, 35:60] = mark
    prepared, mask = isolate_surface(rgb, PROFILE)
    assert mask[40:48, 35:60].all()
    assert prepared[44, 40, 0] < 70
    assert not mask[0, 0]
    assert (prepared[0, 0] == 127).all()


@pytest.mark.parametrize("gain", [0.75, 1.25])
def test_global_exposure_normalization_preserves_clean_appearance(gain):
    rgb = block()
    other = np.clip(np.rint(rgb.astype(float) * gain), 0, 255).astype(np.uint8)
    expected, _ = isolate_surface(rgb, PROFILE)
    actual, _ = isolate_surface(other, PROFILE)
    assert np.abs(expected.astype(float) - actual).max() <= 2


@pytest.mark.parametrize("failure", ["missing", "multiple", "clipped", "tiny"])
def test_unusable_surface_requires_new_view(failure):
    rgb = block()
    if failure == "missing":
        rgb[:] = 80
    elif failure == "multiple":
        rgb[:] = 80
        rgb[10:40, 10:40] = [20, 90, 180]
        rgb[55:85, 55:85] = [20, 90, 180]
    elif failure == "clipped":
        rgb[20:76, :30] = [20, 90, 180]
    else:
        rgb[:] = 80
        rgb[40:50, 40:50] = [20, 90, 180]
    with pytest.raises(SurfaceViewError):
        isolate_surface(rgb, PROFILE)


def test_large_background_hotspot_is_not_a_block_score():
    _, mask = surface_input(block(), (128, 128), PROFILE)
    amap = np.full(mask.shape, 999.0, dtype=np.float32)
    amap[mask] = 2
    score, visual = surface_score(999, amap, mask)
    assert score == 2
    assert not visual[~mask].any()
    with pytest.raises(ValueError, match="anomaly map"):
        surface_score(999, None, mask)


def test_dataset_torch_and_openvino_paths_share_surface_contract(tmp_path):
    import torch

    from brickcell.data.manifest import Manifest, Sample
    from brickcell.detector.dataset import ManifestDataset
    from brickcell.detector.runtime import OpenVINODetector
    from brickcell.detector.workflow import torch_predict

    rgb = block()
    path = tmp_path / "input.png"
    Image.fromarray(rgb).save(path)
    manifest = Manifest(tmp_path / "manifest.json", [Sample("one", "input.png", "good", "train")])
    expected, mask = surface_input(rgb, (128, 128), PROFILE)
    dataset = ManifestDataset(manifest, "train", (128, 128), PROFILE)
    np.testing.assert_array_equal(dataset[0].image.numpy(), expected[0])
    amap = np.full((128, 128), 900, dtype=np.float32)
    amap[mask] = 3

    def model(tensor):
        np.testing.assert_array_equal(tensor.numpy(), expected)
        return SimpleNamespace(pred_score=torch.tensor(900), anomaly_map=torch.from_numpy(amap))

    assert torch_predict(model, rgb, (128, 128), PROFILE)[0] == 3

    def infer(inputs):
        np.testing.assert_array_equal(inputs["input"], expected)
        return {"score": np.array([900]), "map": amap}

    detector = object.__new__(OpenVINODetector)
    detector.surface_profile, detector.size = PROFILE, (128, 128)
    detector.compiled = SimpleNamespace(input=lambda index: "input")
    detector.request = SimpleNamespace(infer=infer)
    detector.outputs = {"pred_score": "score", "anomaly_map": "map"}
    prediction = detector.predict(rgb)
    assert prediction.score == 3
    assert not prediction.anomaly_map[~mask].any()


def test_service_maps_missing_object_to_reinspect(tmp_path):
    from brickcell.inspection.service import InspectionService
    from brickcell.inspection.threshold import Threshold
    from brickcell.sources.directory import Frame

    path = tmp_path / "empty.png"
    Image.fromarray(np.full((96, 96, 3), 80, dtype=np.uint8)).save(path)

    def predict(rgb):
        isolate_surface(rgb, PROFILE)
        pytest.fail("An empty crop cannot produce a score")

    detector = SimpleNamespace(
        surface_profile=PROFILE,
        model_version="test",
        threshold=Threshold(1, 0, "v", "h", 5),
        predict=predict,
    )
    result = InspectionService(detector).inspect(Frame("block", "f", "now", "front", path))
    assert result.decision == "REINSPECT"
    assert not result.is_valid_view
    assert result.anomaly_score is None


def test_invalid_profiles_do_not_silently_use_defaults():
    with pytest.raises(ValueError):
        SurfaceProfile(hue_ranges=[[170, 10]])
    with pytest.raises(ValueError):
        SurfaceProfile(hue_ranges=[[95, 115]], version="different")


def test_blurred_or_overexposed_object_does_not_become_an_anomaly():
    with pytest.raises(SurfaceViewError, match="texture"):
        isolate_surface(block(), {**PROFILE, "min_laplacian_variance": 5})
    rgb = block()
    rgb[20:76, 22:74] = [30, 130, 255]
    with pytest.raises(SurfaceViewError, match="glare"):
        isolate_surface(rgb, PROFILE)


def test_dark_red_cast_background_is_excluded_but_enclosed_black_mark_remains():
    rgb = np.full((96, 96, 3), (29, 18, 15), dtype=np.uint8)
    rgb[20:76, 22:74] = [180, 25, 25]
    rgb[40:46, 38:58] = 0
    profile = {
        "hue_ranges": [[0, 8], [170, 179]],
        "min_saturation": 90,
        "min_value": 40,
        "min_laplacian_variance": 0,
    }
    image, mask = isolate_surface(rgb, profile)
    assert not mask[:20].any()
    assert mask[40:46, 38:58].all()
    assert (image[40:46, 38:58] == 0).all()


def test_pale_block_on_saturated_backdrop_keeps_black_cross():
    rgb = np.full((96, 96, 3), (0, 170, 240), dtype=np.uint8)
    rgb[20:76, 22:74] = [200, 220, 230]
    # A cyan cast on the front face is still less saturated than the plate.
    rgb[48:76, 22:74] = [30, 160, 200]
    rgb[50:64, 45:49] = 0
    rgb[55:59, 37:57] = 0
    profile = dict(version="pale_block_v1", hue_ranges=[], min_value=90, max_saturation=245)
    prepared, mask = isolate_surface(rgb, profile)
    assert not mask[0, 0]
    assert mask[50:64, 45:49].all()
    assert mask[55:59, 37:57].all()
    assert (prepared[55:59, 37:57] == 0).all()
    assert mask[70, 40]


def test_pale_profile_rejects_ambiguous_contract():
    with pytest.raises(ValueError, match="not hue ranges"):
        SurfaceProfile(version="pale_block_v1", hue_ranges=[[0, 8]])
    with pytest.raises(ValueError, match="maximum saturation"):
        SurfaceProfile(version="pale_block_v1", hue_ranges=[], max_saturation=255)
