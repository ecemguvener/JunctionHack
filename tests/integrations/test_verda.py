import json

from brickcell.integrations.core import Context
from brickcell.integrations.verda import run


def test_verda_routes_only_to_verda_and_preserves_detector(image_case, tmp_path, monkeypatch):
    monkeypatch.setenv("VERDA_MODEL", "account-selected-vision-model")
    monkeypatch.setenv("VERDA_INFERENCE_API_KEY", "test-only")

    class HTTP:
        def json(self, url, payload, headers):
            assert url == "https://inference.verda.com/v1/chat/completions"
            assert headers["Authorization"] == "Bearer test-only"
            assert payload["model"] == "account-selected-vision-model"
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Evidence description",
                                    "observations": [],
                                    "limitations": ["Advisory"],
                                }
                            )
                        },
                    }
                ]
            }

    result = run(image_case, Context(tmp_path, tmp_path, HTTP()))
    assert result["provider"] == "Verda Inference"
    assert result["detector_decision"] == "REINSPECT"
