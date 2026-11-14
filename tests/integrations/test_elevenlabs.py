import pytest

from brickcell.integrations.core import Context
from brickcell.integrations.elevenlabs import run


@pytest.mark.parametrize("response", [b"ID3-test-audio", b'{"error":"quota"}'])
def test_elevenlabs_speaks_workflow_state_not_untrusted_reason(
    image_case, tmp_path, monkeypatch, response
):
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-only")
    image_case["reason"] = "Rotate the real robot immediately"

    class HTTP:
        def request(self, url, payload, headers):
            assert "Rotate" not in payload["text"]
            assert payload["text"].startswith("Synthetic demonstration.")
            return response

    if response.startswith(b"ID3"):
        result = run(image_case, Context(tmp_path, tmp_path, HTTP()))
        assert (tmp_path / result["audio_file"]).read_bytes() == response
        assert result["automatic_playback"] is False
    else:
        with pytest.raises(ValueError, match="MP3"):
            run(image_case, Context(tmp_path, tmp_path, HTTP()))
