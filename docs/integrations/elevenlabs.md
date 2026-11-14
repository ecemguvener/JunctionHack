# ElevenLabs — spoken operator guidance

ElevenLabs converts a fixed, decision-specific operator message to MP3. It speaks
REINSPECT/ERROR as review or pause instructions, not as a defect classification.
Synthetic cases are announced as synthetic. Neither raw case text nor Gemini
output is fed directly to speech, preventing an image prompt or error message
from speaking unverified movement instructions.

Export `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, optionally `ELEVENLABS_MODEL`
(default `eleven_multilingual_v2`). Run the sidecar with `--providers elevenlabs
--execute` plus your case and evidence-root arguments. Receipts include exact
spoken text, voice/model, audio filename and SHA-256. Use the MP3 with an explicit
operator Play button; this integration never starts audio or commands an arm.
The outbox suppresses repeat successful requests. Voice/model/text/case identity
form the audio filename, preserving evidence of what was said for each case.

Offline tests inspect the actual request, synthetic announcement, exclusion of
untrusted reason text and refusal of non-audio responses. Voice availability,
billing, pronunciation and audio playback remain unverified on a live account.
[ElevenLabs create speech API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert).
