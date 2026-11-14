from pathlib import Path

import pytest

from brickcell.data.manifest import load_manifest
from brickcell.data.synthetic import generate


@pytest.fixture
def manifest(tmp_path: Path):
    return load_manifest(generate(tmp_path / "synthetic"))
