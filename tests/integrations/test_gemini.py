import json

import pytest

from brickcell.integrations.core import Context
from brickcell.integrations.gemini import run


@pytest.mark.parametrize("attempt_override", [False, True])
def test_gemini_cannot_promote_detector_decision(
    image_case, tmp_path, monkeypatch, attempt_override
):
    monkeypatch.setenv("GEMINI_MODEL", "account-selected-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")

    class HTTP:
        def json(self, url, payload, headers):
            assert headers["x-goog-api-key"] == "test-only"
            assert payload["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/png"
            value = {
                "summary": "A blue surface is visible.",
                "observations": ["One view"],
                "limitations": ["Synthetic example"],
            }
            if attempt_override:
                value["decision"] = "GOOD"
            return {
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(value)}]}}
                ]
            }

    if attempt_override:
        with pytest.raises(ValueError, match="classification"):
            run(image_case, Context(tmp_path, tmp_path, HTTP()))
    else:
        result = run(image_case, Context(tmp_path, tmp_path, HTTP()))
        assert result["detector_decision"] == "REINSPECT"
        assert result["advisory_only"] is True
    assert image_case["decision"] == "REINSPECT"
