from dataclasses import replace
from pathlib import Path

import pytest

from brickcell.inspection.coverage import FIVE_FACES, CoverageLedger
from brickcell.inspection.types import Decision, InspectionResult
from brickcell.sources.directory import Frame


def observation(face, number=0, decision=Decision.GOOD):
    frame = Frame("brick", f"{face}-{number}", "capture-time", face, Path("unused.png"))
    result = InspectionResult(
        "brick", frame.frame_id, frame.timestamp, face, True, 0.2, decision, "model", "threshold", 1
    )
    return frame, result


def test_good_requires_all_five_faces_and_bottom_is_excluded():
    ledger = CoverageLedger("brick")
    for face in FIVE_FACES[:-1]:
        assert ledger.add(face, *observation(face)).decision == Decision.REINSPECT
    report = ledger.add("top", *observation("top"))
    assert report.decision == Decision.GOOD
    assert not report.missing_faces
    with pytest.raises(ValueError):
        ledger.add("bottom", *observation("bottom"))


def test_defective_face_can_reject_before_complete_coverage():
    ledger = CoverageLedger("brick")
    report = ledger.add("left", *observation("left", decision=Decision.DEFECTIVE))
    assert report.decision == Decision.DEFECTIVE
    assert len(report.missing_faces) == 4
    # A later clean-looking angle must not erase a previously observed defect.
    ledger.add("left", *observation("left", number=1))
    assert ledger.decision().decision == Decision.DEFECTIVE


@pytest.mark.parametrize(
    "changes",
    [
        {"is_valid_view": False},
        {"threshold_version": None},
        {"anomaly_score": None},
        {"decision": Decision.REINSPECT},
    ],
)
def test_invalid_or_uncalibrated_results_do_not_count(changes):
    ledger = CoverageLedger("brick")
    frame, result = observation("front")
    assert ledger.add("front", frame, replace(result, **changes)).decision == Decision.REINSPECT
    assert not ledger.decision().accepted_faces


def test_duplicate_and_wrong_identity_are_rejected():
    ledger = CoverageLedger("brick")
    frame, result = observation("front")
    with pytest.raises(ValueError):
        ledger.add("right", frame, result)
    with pytest.raises(ValueError):
        ledger.add("front", frame, replace(result, object_id="old-brick"))
    ledger.add("front", frame, result)
    with pytest.raises(ValueError):
        ledger.add("front", frame, result)


def test_error_prevents_sort_and_a_new_view_can_resolve_uncertainty():
    ledger = CoverageLedger("brick")
    ledger.add("front", *observation("front", decision=Decision.ERROR))
    assert ledger.decision().decision == Decision.ERROR
    ledger.add("front", *observation("front", number=1))
    assert ledger.decision().decision == Decision.REINSPECT
