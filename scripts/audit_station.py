"""Capture existing HTTP services for offline inspection review. GET only, no arm commands.

Standalone Python 3.12/stdlib tool: can run on Intel without installing Brickcell.
The captured images are new observations, NOT asserted to be the detector's input.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect refused: audit only reads the configured service")


def base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use a loopback HTTP base URL, directly on Intel or through a tunnel")
    return value.rstrip("/")


def get_json(url: str, timeout: float) -> dict:
    started = time.time()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(urllib.request.Request(url, method="GET"), timeout=timeout) as response:
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("Response exceeds 8 MiB limit")
        data = json.loads(body, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return dict(
            url=url,
            ok=True,
            requested_at=started,
            received_at=time.time(),
            sha256=hashlib.sha256(body).hexdigest(),
            data=data,
        )
    except (OSError, ValueError) as exc:
        return dict(
            url=url, ok=False, requested_at=started, received_at=time.time(), error=str(exc)
        )


def collect(
    dashboard="http://127.0.0.1:8096",
    depth="http://127.0.0.1:8094",
    camera_wall="http://127.0.0.1:8093",
    arm="http://127.0.0.1:8765",
    timeout=4.0,
    samples=3,
    interval=0.5,
    run_report: Path | None = None,
) -> dict:
    if not 1 <= samples <= 10 or not 0 <= interval <= 5 or not 0 < timeout <= 10:
        raise ValueError("Require 1–10 samples, 0–5 s interval and 0–10 s timeout")
    dashboard, depth, camera_wall, arm = map(base_url, (dashboard, depth, camera_wall, arm))
    bundle = dict(
        schema_version=1,
        mode="READ_ONLY_STATION_AUDIT",
        started_at=time.time(),
        physical_success_verified=False,
        samples=[],
        note="Camera snapshots are not proven detector input frames; no robot commands issued.",
    )
    endpoints = {
        "dashboard": dashboard + "/api/status",
        "top": depth + "/snapshot/top",
        "plate": depth + "/snapshot/plate",
        "camera_wall": camera_wall + "/api/status",
        "arm_health": arm + "/health",
    }
    with ThreadPoolExecutor(max_workers=5) as pool:
        for index in range(samples):
            futures = {name: pool.submit(get_json, url, timeout) for name, url in endpoints.items()}
            bundle["samples"].append({name: future.result() for name, future in futures.items()})
            if index + 1 < samples:
                time.sleep(interval)
    if run_report:
        try:
            with run_report.open("rb") as stream:
                payload = stream.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                raise ValueError("Run report too large")
            bundle["run_report"] = dict(
                path=str(run_report),
                sha256=hashlib.sha256(payload).hexdigest(),
                data=json.loads(payload),
            )
        except (OSError, ValueError) as exc:
            bundle["run_report"] = dict(path=str(run_report), error=str(exc))
    bundle["finished_at"] = time.time()
    return bundle


def analyze(bundle: dict) -> dict:
    if bundle.get("schema_version") != 1 or not isinstance(bundle.get("samples"), list):
        raise ValueError("Unsupported audit bundle")
    findings = []
    frames = {"top": set(), "plate": set()}
    previous = None
    for index, sample in enumerate(bundle["samples"]):
        for name, response in sample.items():
            if not response.get("ok"):
                findings.append(
                    dict(
                        code="SOURCE_UNAVAILABLE",
                        sample=index,
                        source=name,
                        detail=response.get("error", "Unknown error"),
                    )
                )
        for name in frames:
            response = sample.get(name, {})
            data = response.get("data", {})
            if not response.get("ok"):
                continue
            fid, captured = data.get("frame_id"), data.get("time")
            if not fid or not data.get("serial"):
                findings.append(dict(code="CAPTURE_IDENTITY_MISSING", sample=index, source=name))
            if fid in frames[name]:
                findings.append(dict(code="REPEATED_CAMERA_FRAME", sample=index, source=name))
            if fid:
                frames[name].add(fid)
            if type(captured) not in (float, int) or not math.isfinite(captured):
                findings.append(dict(code="CAPTURE_TIME_MISSING", sample=index, source=name))
            elif not -0.5 <= response["received_at"] - captured <= 3:
                findings.append(dict(code="CAMERA_AGE_OR_CLOCK_SKEW", sample=index, source=name))
        status = sample.get("dashboard", {}).get("data", {})
        for name in ("anomalies", "focus"):
            overlay = status.get(name, {})
            if not overlay.get("connected"):
                findings.append(
                    dict(
                        code="OVERLAY_NOT_CURRENT",
                        sample=index,
                        source=name,
                        detail=overlay.get("reason", "No overlay status"),
                    )
                )
            # The legacy endpoint drops provenance. Even an empty list cannot prove
            # a fresh inspection occurred, nor that an object is GOOD.
            required = ("frame_id", "camera_id", "captured_at", "coordinate_space")
            missing = [k for k in required if overlay.get(k) is None]
            if missing:
                findings.append(
                    dict(
                        code="OVERLAY_UNBOUND_TO_CAPTURE",
                        sample=index,
                        source=name,
                        missing=missing,
                    )
                )
        anomaly = status.get("anomalies", {})
        if anomaly.get("items") and not all(
            anomaly.get(key) for key in ("method", "model_version", "threshold_version")
        ):
            findings.append(dict(code="DETECTOR_PROVENANCE_MISSING", sample=index))
        semantic = {k: anomaly.get(k) for k in ("source", "items")}
        if previous and semantic == previous[0] and anomaly.get("connected"):
            old, new = previous[1], anomaly.get("age_s")
            if type(old) in (float, int) and type(new) in (float, int) and new < old:
                findings.append(
                    dict(
                        code="POSSIBLE_HEARTBEAT_REFRESH",
                        sample=index,
                        detail="Unchanged payload age reset; cannot prove fresh inference.",
                    )
                )
        previous = (semantic, anomaly.get("age_s"))
    run = bundle.get("run_report", {}).get("data", {})
    if isinstance(run, dict) and run.get("succeeded"):
        findings.append(
            dict(
                code="RUN_SUCCESS_IS_SELF_REPORTED",
                detail="Cycle counts do not verify correct grasps, classifications or placements.",
            )
        )
    return dict(
        schema_version=1,
        mode="READ_ONLY_STATION_AUDIT",
        sample_count=len(bundle["samples"]),
        unique_frames={k: len(v) for k, v in frames.items()},
        physical_success_verified=False,
        detection_accuracy_evaluated=False,
        findings=findings,
    )


def render(bundle: dict, summary: dict) -> str:
    def esc(value):
        return html.escape(str(value), quote=True)

    cards = []
    for i, sample in enumerate(bundle["samples"]):
        pictures = []
        for name in ("top", "plate"):
            data = sample.get(name, {}).get("data", {})
            encoded = data.get("rgb_jpeg_b64", "")
            try:
                raw = base64.b64decode(encoded, validate=True)
                if not raw.startswith(b"\xff\xd8"):
                    raise ValueError("Not JPEG")
                picture = f'<img alt="{name} audit capture" src="data:image/jpeg;base64,{encoded}">'
            except (ValueError, TypeError):
                picture = "<p>No valid image captured.</p>"
            pictures.append(
                f"<article><h3>{name.title()} camera</h3>{picture}"
                f"<p>Frame: {esc(data.get('frame_id', 'unavailable'))}<br>"
                f"Serial: {esc(data.get('serial', 'unavailable'))}</p></article>"
            )
        status = sample.get("dashboard", {}).get("data", {})
        overlays = {k: status.get(k) for k in ("focus", "anomalies")}
        cards.append(
            f"<h2>Sample {i + 1}</h2><section>{''.join(pictures)}</section>"
            f"<details><summary>Dashboard result payloads</summary>"
            f"<pre>{esc(json.dumps(overlays, indent=2))}</pre></details>"
        )
    codes = sorted({f["code"] for f in summary["findings"]})
    return (
        '<!doctype html><meta charset="utf-8"><title>Inspection evidence audit</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<style>body{font:16px system-ui;margin:24px;background:#101722;color:#edf2fa}"
        "main{max-width:1200px;margin:auto}section{display:flex;gap:18px;flex-wrap:wrap}"
        "article{flex:1;min-width:280px;background:#202b3a;padding:16px}img{width:100%}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:8px 0}"
        ".notice{border-left:4px solid #f9b64b;padding:16px;background:#302a21}</style><main>"
        '<h1>Inspection evidence audit</h1><p class="notice">Read-only capture. '
        "No robot commands. These images are not proven to be the detector input. "
        "Boxes are deliberately not drawn onto unrelated frames. "
        "A connected publisher does not establish fresh inference or correct detection.</p>"
        f"<p>{summary['sample_count']} samples · {esc(summary['unique_frames'])} unique frames</p>"
        "<h2>Checks requiring attention</h2><ul>"
        + "".join(f"<li>{esc(code)}</li>" for code in codes)
        + "</ul>"
        + "".join(cards)
        + "<h2>Machine-readable assessment</h2><pre>"
        + esc(json.dumps(summary, indent=2))
        + "</pre></main>"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    for name, port in [("dashboard", 8096), ("depth", 8094), ("camera-wall", 8093), ("arm", 8765)]:
        capture.add_argument("--" + name, default=f"http://127.0.0.1:{port}")
    capture.add_argument("--samples", type=int, default=3)
    capture.add_argument("--interval", type=float, default=0.5)
    capture.add_argument("--timeout", type=float, default=4)
    capture.add_argument("--run-report", type=Path)
    capture.add_argument("--stdout", action="store_true", help="Emit bundle only (for SSH capture)")
    capture.add_argument("--output", type=Path)
    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture" and not args.stdout and not args.output:
        parser.error("capture needs --stdout or --output")
    if args.output and args.output.exists():
        parser.error("Choose a new output directory; existing evidence is preserved")
    if args.command == "capture":
        bundle = collect(
            args.dashboard,
            args.depth,
            args.camera_wall,
            args.arm,
            args.timeout,
            args.samples,
            args.interval,
            args.run_report,
        )
        if args.stdout:
            print(json.dumps(bundle, allow_nan=False))
            return 0
    else:
        bundle = json.loads(args.bundle.read_text())
    summary = analyze(bundle)
    args.output.mkdir(parents=True)
    for name, value in [("bundle", bundle), ("summary", summary)]:
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    (args.output / "index.html").write_text(render(bundle, summary))
    print(json.dumps(summary, indent=2))
    print(f"Review: {args.output.resolve() / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
