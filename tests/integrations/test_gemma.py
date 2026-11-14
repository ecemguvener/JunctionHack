import json

import pytest

from brickcell.integrations.core import Context
from brickcell.integrations.gemma import run


def test_gemma_uses_local_multimodal_endpoint_without_openai_key(image_case, tmp_path, monkeypatch):
    monkeypatch.setenv("GEMMA_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("GEMMA_MODEL", "self-hosted-image-capable-gemma")

    class HTTP:
        def json(self, url, payload, headers):
            assert url == "http://127.0.0.1:8000/v1/chat/completions"
            assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png"
            )
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Blue surface",
                                    "observations": [],
                                    "limitations": ["One view only"],
                                }
                            )
                        },
                    }
                ]
            }

    result = run(image_case, Context(tmp_path, tmp_path, HTTP()))
    assert result["provider"] == "Google Gemma (self-hosted)"
    assert result["detector_decision"] == "REINSPECT"
    monkeypatch.setenv("GEMMA_BASE_URL", "https://remote.example/v1")
    with pytest.raises(ValueError, match="authentication"):
        run(image_case, Context(tmp_path, tmp_path, HTTP()))
