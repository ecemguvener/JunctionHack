"""Oxylabs Web API: discover and read approved manufacturer guidance with citations."""

from brickcell.integrations.core import required
from brickcell.integrations.reference import approved_url, save_reference


def run(case, context):
    query = case.get("research_query", "")
    if not isinstance(query, str) or not 1 <= len(query) <= 2048:
        raise ValueError("Supply one bounded research_query; camera images are not sent")
    headers = {"Authorization": "Bearer " + required("OXYLABS_WEB_API_KEY")}
    result = context.http.json(
        "https://webapi.oxylabs.io/v1/search",
        {"query": query, "max_results": 3},
        headers,
        timeout=150,
    )
    if result.get("state") != "done":
        raise ValueError("Oxylabs search did not complete")
    sources = []
    for hit in result.get("results", [])[:3]:
        url = hit.get("url", "")
        try:
            approved_url(url)
            page = context.http.json(
                "https://webapi.oxylabs.io/v1/scrape",
                {"url": url, "output": ["markdown"]},
                headers,
                timeout=150,
            )
            rows = page.get("results", [])
            if page.get("state") != "done" or not rows or not rows[0].get("markdown"):
                raise ValueError("Scrape did not return readable content")
            final = approved_url(rows[0].get("url", url))
            source = save_reference(
                "Oxylabs Web API",
                final,
                {"markdown": rows[0]["markdown"]},
                context,
                page.get("metadata", {}).get("request_id"),
            )
            sources.append(source)
        except Exception as exc:
            # Record source failures instead of answering from truncated search snippets.
            sources.append(
                {
                    "source_url": url,
                    "error_type": type(exc).__name__,
                    "used_for_detector_decision": False,
                }
            )
    if not any("reference_file" in source for source in sources):
        raise ValueError("No approved source could be retrieved")
    return {
        "query": query,
        "sources": sources,
        "provider": "Oxylabs Web API",
        "advisory_only": True,
    }
