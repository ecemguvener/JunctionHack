# Context.dev — structured manufacturer reference records

Context.dev extracts named reference fields from a deliberately selected source
URL using explicit CSS rules. This complements Oxylabs discovery: the operator
chooses a manufacturer page and the fields needed for a report/knowledge record.
The connector preserves the final source URL, retrieval time, request ID and
content hash. Extracted values remain untrusted reference information, separate
from physical measurements, human labels and detector decisions.

Export `CONTEXT_DEV_API_KEY` and exact comma-separated
`REFERENCE_ALLOWED_HOSTS`. Add `reference_url` and optional `reference_rules` to
a case; for example `{"title":"title","material":".material","guidance":"article"}`.
Use selectors verified against the chosen page; a missing selector isn't evidence
of an absent product property. Run the sidecar with `--providers contextdev
--execute`. Only the URL and extraction rules are sent, never camera images.
The current endpoint is `https://api.context.dev/v1/web/scrape`, with
`formats.parse` and `parseParams.rules`; it is not an invented generic LLM API.

Both HTTP success and `parsed.success == true` are required. Failed/empty output,
invalid rules or an unapproved final URL produces a failed receipt. Successful
fields are stored as a bounded, hashed knowledge record under `references/` and
can be reviewed in the case portal. The connector requests fresh output with
`maxAgeMs: 0`, but retrieval time still does not establish when a manufacturer
last updated its page. It does not feed reference text into detector training.

Offline tests verify request structure, source records and refusal of partial
failures returned with HTTP 200. Actual CSS selectors, source truth, API quota
and live extraction remain unverified. [Context.dev custom parser](https://docs.context.dev/scrape/parse-fields).
