from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Frame:
    object_id: str
    frame_id: str
    timestamp: str
    view_id: str
    path: Path


class FrameSource(Protocol):
    def capture(self, object_id: str, attempt: int) -> Frame | None: ...


class DirectorySource:
    """Explicit object -> ordered views mapping. Never silently borrow another object's frame."""

    def __init__(self, frames: dict[str, list[Path]]) -> None:
        self.frames = frames
        self.sequence = 0

    def capture(self, object_id: str, attempt: int) -> Frame | None:
        images = self.frames.get(object_id, [])
        if attempt >= len(images):
            return None
        self.sequence += 1
        return Frame(
            object_id,
            f"{object_id}-frame-{self.sequence}",
            datetime.now(timezone.utc).isoformat(),
            f"view-{attempt}",
            images[attempt],
        )

    @classmethod
    def from_directory(cls, path: Path) -> DirectorySource:
        # One file per object; use a manifest/object mapping for multiple views.
        paths = sorted(
            p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not paths:
            raise ValueError(f"No supported images in {path}")
        return cls({f"object-{i:04d}": [p] for i, p in enumerate(paths)})
