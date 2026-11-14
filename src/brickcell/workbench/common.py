"""Shared report utilities. Reports are evidence, not hardware acceptance."""

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def image_uri(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((480, 360))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def finite_list(array: np.ndarray) -> list:
    return np.where(np.isfinite(array), array, None).tolist()


def fresh_output(path: Path) -> None:
    if path.exists():
        raise ValueError(f"Refusing to overwrite report directory: {path}")
    path.mkdir(parents=True)
