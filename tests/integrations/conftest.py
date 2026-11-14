import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def image_case(tmp_path):
    case = json.loads(Path("configs/integrations/demo-case.json").read_text())
    path = tmp_path / "observed.png"
    Image.new("RGB", (32, 32), "blue").save(path)
    case["assets"] = [
        {
            "role": "observed",
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]
    return case
