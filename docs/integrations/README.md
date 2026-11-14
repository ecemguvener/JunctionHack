# Brickcell optional integrations

These services consume completed inspection cases outside the robot and detector
control paths. They never produce a trusted sorting decision. The existing
RGB-only follower inspection, camera sources and dashboard are untouched.
No provider is enabled automatically and no cloud SDK is added to the core lock.

## One case, separate products

Capture/inspect locally → freeze detector result and evidence hashes → sidecar
case → provider delivery → stored receipt → operator evidence/explanation/audio.
Use `python -m brickcell.integrations` for the sidecar. A default invocation only
prints a plan; `--execute` enables provider calls explicitly. Each product has
its own guide here, environment variables, failure behavior and verification.
Credentials come from the process environment; never put them in case JSON.

```bash
uv run python -m brickcell.integrations \
  --case configs/integrations/demo-case.json --providers cloudinary
```

The included case is SYNTHETIC and REINSPECT, with no images or claimed model.
For a real case, supply the actual object/session/frame/camera identities,
acquisition time in Unix seconds, raw distance, model/threshold versions,
valid-view flag, decision, latency and explicit synthetic flag. Evidence assets
use `role` (reference/observed/overlay), relative `path` and `sha256`. Optional
`localized_region` is a source-image pixel box, never a new detector decision.
The sidecar refuses uncalibrated GOOD/DEFECTIVE cases. It records cases, not
live authorization: old inspection cases are historical evidence, not permission
to move a robot. Do not relabel reference-only demonstrations as calibrated.

SQLite WAL receipts key deliveries by immutable case hash + provider. Successful
deliveries are skipped on repeat runs; failed ones retry on the next explicit
execution. This is a single-worker outbox. A crash after remote success and before
local receipt can cause redelivery; consumers must deduplicate by case ID.
Do not run concurrent sidecar workers against one outbox. Exceptions are recorded
by type only to avoid leaking database URLs or credentials into operator logs.
Provider outages do not enter detector/controller execution or change results.

Evidence paths must stay within `--evidence-root`, match hashes, be PNG/JPEG and
be at most 10 MB. HTTP uses TLS verification, refuses redirects and userinfo URLs,
and has bounded timeouts and response sizes. Loopback HTTP is for local models.
Do not expose local credentials in browser code. Cloud accounts, boards and hosted
compute need configuration before live verification; offline contract tests are
not proof that an account is connected, deployed, paid for or physically working.
