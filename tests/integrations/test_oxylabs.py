import json

from brickcell.integrations.core import Context
from brickcell.integrations.oxylabs import run


def test_oxylabs_reads_approved_page_instead_of_using_search_snippet(
    image_case, tmp_path, monkeypatch
):
    monkeypatch.setenv("OXYLABS_WEB_API_KEY", "test-only")
    monkeypatch.setenv("REFERENCE_ALLOWED_HOSTS", "manufacturer.example")
    image_case["research_query"] = "manufacturer brick material inspection guidance"

    class HTTP:
        def json(self, url, payload, headers, timeout):
            assert timeout == 150
            assert "assets" not in payload
            if url.endswith("/search"):
                return {
                    "state": "done",
                    "results": [
                        {
                            "url": "https://manufacturer.example/guide",
                            "outline": ["Unverified snippet"],
                        },
                        {"url": "http://127.0.0.1/private"},
                    ],
                }
            return {
                "state": "done",
                "results": [{"url": payload["url"], "markdown": "# Published guidance"}],
                "metadata": {"request_id": "vendor-request-1"},
            }

    result = run(image_case, Context(tmp_path, tmp_path, HTTP()))
    record = json.loads((tmp_path / result["sources"][0]["reference_file"]).read_text())
    assert record["content"] == {"markdown": "# Published guidance"}
    assert record["used_for_detector_decision"] is False
    assert result["sources"][1]["error_type"] == "ValueError"
