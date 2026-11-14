"""Context.dev: structured manufacturer reference extraction with source hashes."""

from brickcell.integrations.core import canonical, required
from brickcell.integrations.reference import approved_url, save_reference


def run(case, context):
    url = approved_url(case.get("reference_url", ""))
    rules = case.get("reference_rules", {"title": "title", "heading": "h1", "body": "main"})
    if not isinstance(rules, dict) or not 1 <= len(rules) <= 100 or len(canonical(rules)) > 20_000:
        raise ValueError("Use bounded explicit CSS extraction rules")
    response = context.http.json(
        "https://api.context.dev/v1/web/scrape",
        {
            "url": url,
            "formats": {"parse": True},
            "parseParams": {"type": "custom", "rules": rules},
            "maxAgeMs": 0,
        },
        {"Authorization": "Bearer " + required("CONTEXT_DEV_API_KEY")},
        timeout=60,
    )
    parsed = response.get("parsed", {})
    if (
        parsed.get("success") is not True
        or not isinstance(parsed.get("data"), dict)
        or not any(parsed["data"].values())
    ):
        raise ValueError("Context.dev returned failed or empty structured extraction")
    source = approved_url(response.get("url", url))
    record = save_reference(
        "Context.dev", source, parsed["data"], context, response.get("request_id")
    )
    return {
        "provider": "Context.dev",
        "fields": parsed["data"],
        "reference": record,
        "extraction_method": "explicit_css_rules",
        "advisory_only": True,
    }
