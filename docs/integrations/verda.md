# Verda — hosted inference for optional evidence explanations

Verda Inference is an alternative hosted compute service for the visual evidence
assistant. The adapter calls Verda's documented `/v1/chat/completions` protocol;
it does not send data to OpenAI or use an OpenAI key. It sends hash-verified
images and the same restricted explanation schema as the other assistants.
The returned receipt identifies Verda, the selected model and source hashes.
Local PyTorch training and OpenVINO CPU anomaly inference remain unchanged.

Export `VERDA_INFERENCE_API_KEY` and `VERDA_MODEL`. Select an image-capable model
currently available in the account that supports JSON-schema structured outputs.
A text-only endpoint is insufficient. Run the sidecar with `--providers verda
--execute` plus the actual case and evidence-root. If the chosen model doesn't
support images/schema, the request fails visibly; there is no silent text-only
fallback or fabricated result. Verda's Cloud API provisioning credentials are
not the same as its Inference API key.

This feature uses hosted inference, not automatic GPU-instance provisioning or
PatchCore training. It deliberately avoids creating billable instances. For
larger training, use the separately isolated Modal job or provision your own
Verda environment and preserve Brickcell's explicit dataset groups/contracts.
Do not interpret an explanation response as detector calibration or hardware
performance. Vision/schema availability, account quota and live response quality
remain unverified; offline tests verify endpoint/authentication/model routing and
preserved decisions. [Verda inference quickstart](https://docs.verda.com/inference/quickstart/).
