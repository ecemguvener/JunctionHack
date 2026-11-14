# Google Gemini API — grounded visual explanation

Gemini receives up to three hash-verified images, their roles, the detector's raw
distance, decision, model/threshold provenance and optional localized region.
Its structured JSON describes visible features and ambiguity. It cannot supply
a classification: unexpected response fields (including `decision`) are rejected.
The receipt preserves the original detector decision. The next operator action
comes from a fixed local mapping, never free-form model instructions. Image text
is treated as untrusted input. This boundary prevents a fluent explanation from
becoming a robot command or evidence of unseen-face coverage.

Export `GEMINI_API_KEY` and `GEMINI_MODEL` (a currently available image-capable
model ID in your account). No OpenAI account or key is used. Run:

```bash
uv run python -m brickcell.integrations --case artifacts/case.json \
  --providers gemini --evidence-root artifacts --execute
```

The JSON receipt contains the summary, observations, limitations, source hashes,
provider/model and deterministic next step. Blocked, truncated, invalid JSON or
unexpected fields produce a failed receipt; no replacement explanation is
invented. Read the limitations with the summary. Language models can still
misdescribe evidence despite the schema; operators must review the images.

Offline tests verify real request image encoding and reject attempted decision
overrides. Live image understanding, account quota and model availability are
not yet verified. [Google structured outputs](https://ai.google.dev/gemini-api/docs/structured-output).
