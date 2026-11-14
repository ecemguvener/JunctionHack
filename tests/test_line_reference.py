import numpy as np
import pytest

from brickcell.inspection.line_reference import localize


def example():
    rgb = np.full((48, 48, 3), 30, np.uint8)
    rgb[8:40, 8:40] = [20, 160, 220]
    mask = np.zeros((48, 48), bool)
    mask[8:40, 8:40] = True
    return rgb, mask


def test_reference_mark_is_localized_without_a_trained_model():
    clean, mask = example()
    changed = clean.copy()
    changed[22:25, 18:30] = [10, 70, 100]
    result, _, _ = localize(clean, changed, mask, mask)
    assert result["status"] == "DARK_MARK_CANDIDATE"
    assert result["boxes"][0]["box"] == [18, 22, 30, 25]
    assert result["calibrated"] is False


def test_clean_reference_does_not_invent_mark():
    clean, mask = example()
    result, _, _ = localize(clean, clean, mask, mask)
    assert not result["boxes"]


def test_large_motion_requests_new_reference():
    clean, mask = example()
    with pytest.raises(ValueError, match="moved too far"):
        localize(clean, clean, mask, np.roll(mask, 5, axis=0))
