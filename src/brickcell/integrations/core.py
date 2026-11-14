"""Immutable inspection cases, bounded HTTP and persistent delivery receipts."""

import base64
import hashlib
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class IntegrationError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise IntegrationError(f"Missing environment variable: {name}")
    return value


def validate(case):
    for key in ("object_id", "session_id", "frame_id", "camera_id", "model_version"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"Case needs {key}")
    if case.get("decision") not in ("GOOD", "DEFECTIVE", "REINSPECT", "ERROR"):
        raise ValueError("Invalid detector decision")
    if type(case.get("synthetic")) is not bool:
        raise ValueError("Explicit synthetic flag required")
    if case["decision"] in ("GOOD", "DEFECTIVE") and (
        case.get("is_valid_view") is not True or not case.get("threshold_version")
    ):
        raise ValueError("A definite decision requires a valid view and calibrated threshold")
    for name in ("captured_at", "inference_latency_ms"):
        value = case.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Case needs finite nonnegative {name}")
    if case.get("score") is not None and (
        type(case["score"]) not in (int, float) or not math.isfinite(case["score"])
    ):
        raise ValueError("Score must be a finite raw anomaly distance")
    for asset in case.get("assets", []):
        if asset.get("role") not in ("reference", "observed", "overlay"):
            raise ValueError("Invalid evidence role")
        if not isinstance(asset.get("sha256"), str) or len(asset["sha256"]) != 64:
            raise ValueError("Evidence needs SHA-256")
    canonical(case)
    return case


def case_id(case):
    validate(case)
    return hashlib.sha256(canonical(case).encode()).hexdigest()


def evidence(asset, root):
    root = Path(root).resolve()
    path = (root / asset["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 10_000_000:
        raise ValueError("Evidence must be a bounded file inside the evidence root")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != asset["sha256"]:
        raise ValueError("Evidence hash changed")
    from PIL import Image

    import io

    with Image.open(io.BytesIO(content)) as image:
        if image.format not in ("JPEG", "PNG"):
            raise ValueError("Only JPEG/PNG evidence supported")
        image.verify()
        mime = "image/jpeg" if image.format == "JPEG" else "image/png"
    return content, mime


def data_url(asset, root):
    content, mime = evidence(asset, root)
    return f"data:{mime};base64," + base64.b64encode(content).decode()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise IntegrationError("Provider redirects are refused")


class HTTP:
    def request(self, url, payload=None, headers=None, raw=None, timeout=20):
        parts = urlsplit(url)
        if (
            parts.username
            or parts.password
            or not (
                parts.scheme == "https"
                or (parts.scheme == "http" and parts.hostname in ("127.0.0.1", "localhost"))
            )
        ):
            raise IntegrationError("Use HTTPS or a loopback development endpoint")
        headers = dict(headers or {})
        if payload is not None:
            raw = canonical(payload).encode()
            headers.setdefault("Content-Type", "application/json")
        request = Request(url, data=raw, headers=headers)
        try:
            with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
                content = response.read(16_000_001)
                if len(content) > 16_000_000:
                    raise IntegrationError("Provider response exceeded size limit")
                return content
        except HTTPError as exc:
            raise IntegrationError(f"Provider HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise IntegrationError("Provider transport failed") from None

    def json(self, url, payload=None, headers=None):
        return json.loads(self.request(url, payload, headers))


@dataclass
class Context:
    root: Path
    output: Path
    http: HTTP


class Outbox:
    """One local serial worker; remote delivery can be at-least-once after a crash."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=5)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS deliveries (case_id TEXT, provider TEXT, "
            "case_json TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER DEFAULT 0, "
            "receipt TEXT, updated REAL, PRIMARY KEY(case_id, provider))"
        )
        self.db.commit()

    def submit(self, case, providers):
        identifier = case_id(case)
        with self.db:
            for provider in providers:
                self.db.execute(
                    "INSERT OR IGNORE INTO deliveries VALUES (?, ?, ?, 'pending', 0, NULL, ?)",
                    (identifier, provider, canonical(case), time.time()),
                )
        return identifier

    def deliver(self, case, provider, context, run):
        identifier = self.submit(case, [provider])
        row = self.db.execute(
            "SELECT status,receipt FROM deliveries WHERE case_id=? AND provider=?",
            (identifier, provider),
        ).fetchone()
        if row[0] == "done":
            return json.loads(row[1])
        before = canonical(case)
        try:
            # Pass an isolated copy: providers never hold the caller's decision object.
            result = run(json.loads(before), context)
            if not isinstance(result, dict):
                raise IntegrationError("Provider receipt must be an object")
            receipt = {"provider": provider, "case_id": identifier, "result": result}
            encoded, status = canonical(receipt), "done"
        except Exception as exc:
            # Exception text can contain database URLs or provider secrets.
            receipt = {
                "provider": provider,
                "case_id": identifier,
                "error_type": type(exc).__name__,
            }
            encoded, status = canonical(receipt), "failed"
        with self.db:
            self.db.execute(
                "UPDATE deliveries SET status=?,attempts=attempts+1,receipt=?,updated=? "
                "WHERE case_id=? AND provider=?",
                (status, encoded, time.time(), identifier, provider),
            )
        return receipt

    def close(self):
        self.db.close()
