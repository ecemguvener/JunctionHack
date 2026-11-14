import json

import pytest

from brickcell.integrations.contextdev import run
from brickcell.integrations.core import Context


@pytest.mark.parametrize("success", [True, False])
def test_contextdev_checks_output_status_even_for_http_200(
    image_case, tmp_path, monkeypatch, success
):
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "test-only")
    monkeypatch.setenv("REFERENCE_ALLOWED_HOSTS", "manufacturer.example")
    image_case["reference_url"] = "https://manufacturer.example/material"
    image_case["reference_rules"] = {"material": ".material"}

    class HTTP:
        def json(self, url, payload, headers, timeout):
            assert url == "https://api.context.dev/v1/web/scrape"
            assert payload["parseParams"]["rules"] == {"material": ".material"}
            assert payload["maxAgeMs"] == 0
            return {
                "url": image_case["reference_url"],
                "request_id": "request-1",
                "parsed": {"success": success, "data": {"material": "ABS"} if success else None},
            }

    if success:
        result = run(image_case, Context(tmp_path, tmp_path, HTTP()))
        saved = json.loads((tmp_path / result["reference"]["reference_file"]).read_text())
        assert saved["source_url"] == image_case["reference_url"]
        assert saved["untrusted_reference"] is True
        assert result["fields"] == {"material": "ABS"}
    else:
        with pytest.raises(ValueError, match="failed or empty"):
            run(image_case, Context(tmp_path, tmp_path, HTTP()))
