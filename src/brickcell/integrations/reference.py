"""Cited, untrusted reference records; never training labels or detector decisions."""

import hashlib
import os
import time
from urllib.parse import urlsplit

from brickcell.integrations.core import canonical


def approved_url(url):
    allowed = {
        host.strip().lower()
        for host in os.environ.get("REFERENCE_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    }
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.username or parts.password or parts.hostname not in allowed:
        raise ValueError("Reference URL must use HTTPS on an explicitly approved host")
    return url


def save_reference(provider, url, content, context, request_id=None):
    approved_url(url)
    text = canonical(content)
    if len(text.encode()) > 200_000:
        raise ValueError("Reference extraction exceeded bounded knowledge record size")
    digest = hashlib.sha256(text.encode()).hexdigest()
    record = {
        "provider": provider,
        "source_url": url,
        "retrieved_at": time.time(),
        "request_id": request_id,
        "content_sha256": digest,
        "content": content,
        "untrusted_reference": True,
        "used_for_detector_decision": False,
    }
    directory = context.output / "references"
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(canonical([provider, url, digest]).encode()).hexdigest() + ".json"
    path = directory / name
    if not path.exists():
        with path.open("x") as stream:
            stream.write(canonical(record))
    return {
        "reference_file": "references/" + name,
        "source_url": url,
        "content_sha256": digest,
        "used_for_detector_decision": False,
    }
