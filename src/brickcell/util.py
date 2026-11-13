"""Small serialization helpers shared by CLI and adapters."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (Path, Enum)):
        return str(value) if isinstance(value, Path) else value.value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=json_default, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
