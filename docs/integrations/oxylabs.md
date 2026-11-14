# Oxylabs Web API — cited inspection reference research

Oxylabs searches for manufacturer/material/inspection guidance and then scrapes
up to three approved sources as Markdown. This supplies cited background for an
operator report; it does not detect the brick's line, create defect labels or
change the anomaly threshold. Camera images are not sent. Search snippets are
never substituted for reading the source. Redirected sources are rechecked
against the approved hostname list before being stored.

Export `OXYLABS_WEB_API_KEY` for the **Web API** product and
`REFERENCE_ALLOWED_HOSTS` as comma-separated exact manufacturer/documentation
hostnames. Set `research_query` in the case to one specific question, then run
the sidecar with `--providers oxylabs --execute`. A different Oxylabs product's
credentials will not work. Requests use the current Web API `/v1/search` and
`/v1/scrape` endpoints, not the older scraper API authentication contract.

Each reference stores source URL, retrieval time, provider request ID, extracted
content hash and untrusted-reference status. It lives under the sidecar's local
`references/` directory, never the training manifest. Per-source failures remain
visible when another source succeeds; all-source failure produces a failed
receipt. Only explicitly approved HTTPS hosts are fetched through the provider.
The 150-second provider timeout accommodates page rendering while remaining
outside inspection/robot deadlines. There are no background or unbounded retries;
fix authentication/request errors before an explicit rerun. Review source text
before using it in reports; scraped instructions cannot authorize robot actions.

Offline tests cover actual search→scrape payloads, source provenance and blocked
local URLs. No paid search, scrape, manufacturer fact or live quota was verified.
[Oxylabs Web API reference](https://developers.oxylabs.io/products/web-api/for-agents.md).
