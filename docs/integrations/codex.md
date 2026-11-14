# OpenAI Codex — documented engineering contribution

Codex is the development tool used in this session to implement optional service
adapters, provenance boundaries, offline tests, documentation and feature commits.
It is not an inspection runtime dependency, a detector or an extra API that receives
camera images. No OpenAI API key is required by this feature. Git author names are
not changed to imply AI authorship; attribution distinguishes the developer from
the tool used. No claim is made about unrelated historical commits.

Generate a reviewable source/history manifest:

```bash
uv run python scripts/codex_contribution_report.py \
  --output artifacts/integrations/codex-contribution.json
```

The report records actual HEAD/branch, source-file hashes and recent Git history.
It labels test results absent instead of fabricating success, and cannot prove
physical performance. For the demo, show the per-product commit descriptions,
provider contract tests and `docs/VALIDATION.md` alongside this manifest. Include
actual test output from the checked-out commit rather than treating this manifest
as a validation report. The report reads tracked source paths only, not credentials,
private captures or model binaries. Run it with clean tracked sources for a
reproducible commit snapshot; local ignored artifacts are deliberately excluded.
