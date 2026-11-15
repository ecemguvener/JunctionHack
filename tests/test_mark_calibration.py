"""Synthetic score cases check decision policy, not real mark-detection accuracy."""

import pytest

from brickcell.inspection.threshold import calibrate
from brickcell.inspection.types import Decision
from brickcell.metrics.report import decision_metrics


def test_recall_policy_catches_low_score_defect_that_f1_trades_away():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.85, 2.0, 3.0]
    labels = [0] * 10 + [1] * 3
    default = calibrate(scores, labels, "validation")
    assert default.decide(0.85) == Decision.GOOD
    strict = calibrate(
        scores, labels, "validation", minimum_defect_recall=1, maximum_good_nonaccept_rate=0.2
    )
    assert all(strict.decide(score) == Decision.DEFECTIVE for score in scores[-3:])
    assert sum(strict.decide(score) != Decision.GOOD for score in scores[:10]) == 2
    assert strict.method == "validation_recall_constrained"


def test_unseparable_validation_refuses_approval():
    with pytest.raises(ValueError, match="No threshold approved"):
        calibrate(
            [0.1, 0.2, 0.8, 0.2, 0.9],
            [0, 0, 0, 1, 1],
            "validation",
            minimum_defect_recall=1,
            maximum_good_nonaccept_rate=0.1,
        )


def test_uncertainty_does_not_count_toward_required_recall():
    scores, labels = [0.1, 0.2, 0.5, 0.9], [0, 0, 1, 1]
    threshold = calibrate(scores, labels, "validation", 0.1, minimum_defect_recall=1)
    assert threshold.uncertainty_margin > 0
    assert [threshold.decide(s) for s in scores] == [
        Decision.GOOD,
        Decision.GOOD,
        Decision.DEFECTIVE,
        Decision.DEFECTIVE,
    ]


def test_excessive_reinspection_cannot_hide_poor_clean_performance():
    with pytest.raises(ValueError, match="No threshold approved"):
        calibrate(
            [0.1, 0.45, 0.5, 0.9],
            [0, 0, 1, 1],
            "validation",
            0.2,
            minimum_defect_recall=1,
            maximum_good_nonaccept_rate=0.1,
        )


@pytest.mark.parametrize("recall,nonaccept", [(0, 0.1), (1.1, 0.1), (1, -0.1), (1, 1.1)])
def test_invalid_targets_rejected(recall, nonaccept):
    with pytest.raises(ValueError):
        calibrate(
            [0.1, 0.8],
            [0, 1],
            "validation",
            minimum_defect_recall=recall,
            maximum_good_nonaccept_rate=nonaccept,
        )


def test_decision_report_distinguishes_misses_errors_and_uncertainty():
    report = decision_metrics(
        [0, 0, 0, 1, 1, 1, 1],
        ["GOOD", "DEFECTIVE", "REINSPECT", "GOOD", "DEFECTIVE", "REINSPECT", "ERROR"],
    )
    assert report["defects_accepted_as_good"] == 1
    assert report["defects_not_detected"] == 3
    assert report["defect_detection_recall"] == 0.25
    assert report["good_false_rejection_rate"] == pytest.approx(1 / 3)
    assert report["good_nonaccept_rate"] == pytest.approx(2 / 3)
    assert report["reinspection_count"] == 2
    assert report["error_count"] == 1
    assert decision_metrics([0], ["GOOD"])["defect_detection_recall"] is None
    with pytest.raises(ValueError):
        decision_metrics([1], ["UNKNOWN"])
