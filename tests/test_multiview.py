import pytest

from brickcell.inspection.multiview import classify_burst, fuse_views
from brickcell.inspection.types import Decision


def burst(camera, scores, **kwargs):
    return classify_burst(camera, "brick", scores, 1, calibrated=True, **kwargs)


def test_one_bad_view_is_not_averaged_away():
    views = [burst("a", [0.1] * 10), burst("b", [0.2] * 10), burst("c", [2] * 10)]
    assert fuse_views("brick", ("a", "b", "c"), views)["decision"] == "DEFECTIVE"


def test_no_good_when_missing_uncalibrated_or_unstable():
    good = burst("a", [0.1] * 10)
    assert fuse_views("brick", ("a", "b"), [good])["decision"] == "REINSPECT"
    assert classify_burst("b", "brick", [0.1] * 10, 1).decision == Decision.REINSPECT
    assert burst("b", [0.2, 2] * 5).decision == Decision.REINSPECT
    assert burst("b", [0.1] * 6).decision == Decision.REINSPECT
    assert burst("b", [float("nan")] * 10).decision == Decision.REINSPECT


def test_single_spike_does_not_reject_but_persistent_mark_does():
    assert burst("a", [0.2] * 9 + [4]).decision == Decision.GOOD
    assert burst("a", [2] * 9 + [0.2]).decision == Decision.DEFECTIVE


def test_batches_cannot_mix_objects_or_duplicate_cameras():
    good = burst("a", [0.1] * 10)
    with pytest.raises(ValueError, match="Wrong object"):
        fuse_views("different-brick", ("a",), [good])
    with pytest.raises(ValueError, match="Duplicate"):
        fuse_views("brick", ("a", "b"), [good, good])
    assert fuse_views("brick", ("a",), [good])["decision"] == "GOOD"
