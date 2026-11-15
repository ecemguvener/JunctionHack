"""Evidence for scheduled plate/wrist views; this does not prove physical face coverage."""

from dataclasses import dataclass, field

from brickcell.inspection.types import Decision


@dataclass
class MarkScanEvidence:
    """One instance per held brick; caller supplies fresh, object-matched captures."""

    required: frozenset[tuple[str, int]] = field(
        default_factory=lambda: frozenset(
            [("plate", i) for i in range(7)] + [("wrist", i) for i in (0, 3, 6)]
        )
    )
    observations: dict[tuple[str, int], bool] = field(default_factory=dict, init=False)
    frames: set[tuple[str, str]] = field(default_factory=set, init=False)
    defect_found: bool = field(default=False, init=False)

    def record(self, camera: str, stage: int, frame_id: str, *, valid: bool, flagged: bool):
        key = (camera, stage)
        if key not in self.required:
            raise ValueError("Unexpected camera or scan stage")
        if key in self.observations:
            raise ValueError("Scan stage already recorded")
        if not frame_id or (camera, frame_id) in self.frames:
            raise ValueError("Missing or repeated scan frame identity")
        self.frames.add((camera, frame_id))
        self.observations[key] = valid
        self.defect_found |= valid and flagged

    @property
    def missing(self) -> list[tuple[str, int]]:
        return sorted(key for key in self.required if not self.observations.get(key, False))

    def decision(self) -> Decision:
        if self.defect_found:
            return Decision.DEFECTIVE
        if self.missing:
            return Decision.REINSPECT
        return Decision.GOOD
