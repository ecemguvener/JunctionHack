"""ElevenLabs: bounded operator guidance generated from trusted workflow states."""

import hashlib
import os
from urllib.parse import quote

from brickcell.integrations.core import canonical, case_id, required

MESSAGES = {
    "GOOD": "This view passed the calibrated inspection. Review the remaining required faces.",
    "DEFECTIVE": "The detector flagged an anomaly. Review the saved image and marked region.",
    "REINSPECT": "Inspection needs review. Keep the camera fixed and capture a clear valid view.",
    "ERROR": "Inspection encountered an error. Pause and check the acquisition and model status.",
}


def run(case, context):
    voice = required("ELEVENLABS_VOICE_ID")
    model = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
    text = ("Synthetic demonstration. " if case["synthetic"] else "") + MESSAGES[case["decision"]]
    key = hashlib.sha256(canonical([case_id(case), voice, model, text]).encode()).hexdigest()
    audio = context.http.request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{quote(voice, safe='')}?output_format=mp3_44100_128",
        {"text": text, "model_id": model},
        {"xi-api-key": required("ELEVENLABS_API_KEY"), "Accept": "audio/mpeg"},
    )
    if not (
        audio.startswith(b"ID3") or (len(audio) > 1 and audio[0] == 255 and audio[1] & 224 == 224)
    ):
        raise ValueError("ElevenLabs did not return MP3 audio")
    context.output.mkdir(parents=True, exist_ok=True)
    path = context.output / f"speech-{key}.mp3"
    if path.exists() and path.read_bytes() != audio:
        raise ValueError("Preserve earlier audio; choose a new output directory")
    path.write_bytes(audio)
    return {
        "audio_file": path.name,
        "sha256": hashlib.sha256(audio).hexdigest(),
        "text": text,
        "voice_id": voice,
        "model": model,
        "automatic_playback": False,
        "detector_decision": case["decision"],
    }
