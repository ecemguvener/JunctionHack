# Google Gemma — self-hosted visual inspection assistant

This alternative explanation backend sends the same verified evidence and
restricted JSON schema as Gemini to an image-capable Gemma model hosted by you.
It uses the server's chat-completions compatibility protocol; it does not call
OpenAI or require an OpenAI key. A text-only Gemma model is not suitable.
The model names, weights, license acceptance, serving stack and multimodal
projector must be selected and validated in that separate environment.

Export `GEMMA_BASE_URL` (server base including `/v1`), `GEMMA_MODEL` and, for a
remote endpoint, `GEMMA_API_KEY`. Loopback HTTP is supported for local serving;
remote endpoints require HTTPS and authentication. Run the sidecar with
`--providers gemma --execute` and the actual case/evidence root. A server such as
vLLM must support image messages and `response_format` JSON Schema. If it doesn't,
configure a supported server rather than silently falling back to unstructured
text or dropping the images. No model downloads, GPU installation, server startup
or public endpoint creation occur from this adapter.

The receipt identifies **Google Gemma (self-hosted)** and the exact configured
model. Summaries are advisory; schema checks reject classification overrides and
next steps derive locally from the detector. Human image review remains required.
Choose Gemini or Gemma for a demo to avoid paying for duplicate explanations;
both can be run deliberately for an explanation comparison.

Offline tests check actual image request payloads, local routing, preserved
REINSPECT and remote authentication. No self-hosted model or explanation quality
has been verified. [Gemma documentation](https://ai.google.dev/gemma/docs).
