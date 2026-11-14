"""Google Gemma: self-hosted image explanation using an explicitly configured model."""

import os
from urllib.parse import urlsplit

from brickcell.integrations.core import required
from brickcell.integrations.vision_endpoint import explain


def run(case, context):
    base = required("GEMMA_BASE_URL")
    token = os.environ.get("GEMMA_API_KEY", "")
    if urlsplit(base).hostname not in ("localhost", "127.0.0.1") and not token:
        raise ValueError("Remote Gemma endpoint requires authentication")
    return explain(
        case, context, "Google Gemma (self-hosted)", base, required("GEMMA_MODEL"), token
    )
