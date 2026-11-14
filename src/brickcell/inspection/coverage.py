"""Conservative object-level aggregation for explicit, physically verified face IDs.

This ledger does not infer visibility from a camera name. A presentation/capture
adapter must supply the face ID only after verifying the intended face is visible.
The current single-view simulator does not implement that presentation plan.
"""

from dataclasses import dataclass

from brickcell.inspection.types import Decision, InspectionResult
from brickcell.sources.directory import Frame

FIVE_FACES = ("front", "right", "back", "left", "top")


@dataclass(frozen=True)
class CoverageDecision:
    object_id: str
    decision: Decision
    required_faces: tuple[str, ...]
    accepted_faces: tuple[str, ...]
    missing_faces: tuple[str, ...]
    reason: str


class CoverageLedger:
    def __init__(self, object_id: str, required_faces: tuple[str, ...] = FIVE_FACES):
        if not object_id or not required_faces or len(set(required_faces)) != len(required_faces):
            raise ValueError("Object identity and unique required faces are required")
        self.object_id, self.required_faces = object_id, required_faces
        self.results: dict[str, InspectionResult] = {}
        self.frames: set[str] = set()
        self.defect_found = False

    def add(self, face: str, frame: Frame, result: InspectionResult) -> CoverageDecision:
        if face not in self.required_faces:
            raise ValueError("Unexpected inspection face")
        expected = (self.object_id, frame.frame_id, frame.timestamp, face)
        if (frame.object_id, frame.frame_id, frame.timestamp, frame.view_id) != expected or (
            result.object_id,
            result.frame_id,
            result.timestamp,
            result.view_id,
        ) != expected:
            raise ValueError("Stale or mismatched object/frame/face/timestamp")
        if frame.frame_id in self.frames:
            raise ValueError("A frame cannot establish multiple face observations")
        self.frames.add(frame.frame_id)
        self.results[face] = result
        if (
            result.is_valid_view
            and result.anomaly_score is not None
            and result.threshold_version
            and result.decision == Decision.DEFECTIVE
        ):
            self.defect_found = True
        return self.decision()

    def decision(self) -> CoverageDecision:
        trusted = {
            face: r
            for face, r in self.results.items()
            if r.is_valid_view
            and r.anomaly_score is not None
            and r.threshold_version
            and r.decision in (Decision.GOOD, Decision.DEFECTIVE)
        }
        accepted = tuple(face for face in self.required_faces if face in trusted)
        missing = tuple(face for face in self.required_faces if face not in trusted)
        if any(r.decision == Decision.ERROR for r in self.results.values()):
            decision, reason = Decision.ERROR, "Inspection error; do not sort automatically"
        elif self.defect_found:
            decision, reason = Decision.DEFECTIVE, "Defect found on an accepted face"
        elif missing:
            decision, reason = Decision.REINSPECT, "Required faces lack valid calibrated results"
        else:
            decision, reason = Decision.GOOD, "All required faces have accepted GOOD results"
        return CoverageDecision(
            self.object_id, decision, self.required_faces, accepted, missing, reason
        )
