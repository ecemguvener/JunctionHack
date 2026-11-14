import base64
import hashlib
import io
import json
from urllib.parse import parse_qs

from PIL import Image

from brickcell.integrations.cloudinary import run
from brickcell.integrations.core import Context, case_id


def test_cloudinary_uploads_exact_hashed_pixels_without_overwrite(monkeypatch, tmp_path):
    for name in ("CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"):
        monkeypatch.setenv(name, "example")
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(stream, "PNG")
    content = stream.getvalue()
    (tmp_path / "observed.png").write_bytes(content)
    case = json.loads(open("configs/integrations/demo-case.json").read())
    case["assets"] = [
        {"role": "observed", "path": "observed.png", "sha256": hashlib.sha256(content).hexdigest()}
    ]

    class HTTP:
        def request(self, url, headers, raw):
            form = parse_qs(raw.decode())
            assert base64.b64decode(form["file"][0].split(",")[1]) == content
            assert form["overwrite"] == ["false"]
            assert form["type"] == ["authenticated"]
            assert case_id(case) in form["public_id"][0]
            return json.dumps(
                {
                    "public_id": form["public_id"][0],
                    "secure_url": "https://res.cloudinary.com/example/evidence.png",
                }
            ).encode()

    result = run(case, Context(tmp_path, tmp_path, HTTP()))
    assert result["training_files_modified"] is False
    assert result["evidence"][0]["requires_signed_delivery"] is True
    assert (tmp_path / "observed.png").read_bytes() == content
