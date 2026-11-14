"""Authenticated, loopback-bound report service suitable for hosting on Vultr."""

import argparse
import base64
import hmac
import json
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from brickcell.integrations.__main__ import PROVIDERS
from brickcell.integrations.core import Outbox, required, validate

PAGE = r"""<!doctype html><meta charset="utf-8"><title>Brickcell inspection cases</title>
<style>body{background:#101722;color:#e8eff9;font:17px system-ui;max-width:1100px;margin:36px auto;padding:20px}article{background:#1a293b;padding:24px;margin:18px 0;border-radius:12px}pre{white-space:pre-wrap}img{max-width:280px;margin:8px}a{color:#8bcaff}audio{display:block;margin:12px}</style>
<h1>Brickcell · inspection evidence</h1><p>Recorded detector results. Explanations are advisory. This page cannot control a robot.</p><div id="cases"></div>
<script>
async function refresh(){const response=await fetch('/api/cases');if(!response.ok)return;const records=await response.json();const root=document.querySelector('#cases');root.replaceChildren();for(const item of records){const card=document.createElement('article');const heading=document.createElement('h2');heading.textContent=item.case.object_id+' · '+item.case.decision+(item.case.synthetic?' · SYNTHETIC':'');card.append(heading);const meta=document.createElement('p');meta.textContent=item.case.camera_id+' · '+item.case.model_version+' · '+new Date(item.case.captured_at*1000).toISOString();card.append(meta);for(const delivery of item.deliveries){const text=document.createElement('pre');text.textContent=delivery.provider+' · '+delivery.status+'\n'+JSON.stringify(delivery.receipt,null,2);card.append(text);const result=delivery.receipt?.result;if(result?.evidence){for(const asset of result.evidence){if(!asset.requires_signed_delivery&&asset.url.startsWith('https://res.cloudinary.com/')){const img=document.createElement('img');img.src=asset.url;img.alt=asset.role+' · '+asset.sha256;card.append(img)}}}if(result?.audio_file&&/^speech-[a-f0-9]{64}\.mp3$/.test(result.audio_file)){const audio=document.createElement('audio');audio.controls=true;audio.src='/audio/'+result.audio_file;card.append(audio)}}root.append(card)}}refresh();setInterval(refresh,5000);
</script>"""


def records(database):
    if not database.exists():
        return []
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as db:
        identifiers = db.execute(
            "SELECT case_id,MAX(updated) FROM deliveries GROUP BY case_id ORDER BY MAX(updated) DESC LIMIT 100"
        ).fetchall()
        output = []
        for identifier, _ in identifiers:
            rows = db.execute(
                "SELECT case_json,provider,status,receipt FROM deliveries WHERE case_id=?",
                (identifier,),
            ).fetchall()
            output.append(
                {
                    "case_id": identifier,
                    "case": json.loads(rows[0][0]),
                    "deliveries": [
                        {"provider": p, "status": s, "receipt": json.loads(r) if r else None}
                        for _, p, s, r in rows
                    ],
                }
            )
        return output


def handler(output, username, password):
    expected = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if code == 401:
                self.send_header("WWW-Authenticate", 'Basic realm="Brickcell cases"')
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), expected)

        def do_GET(self):
            if self.path == "/health":
                self.send(200, b'{"service":"brickcell-reports","robot_control":false}')
                return
            if not self.authorized():
                self.send(401, b'{"error":"Authentication required"}')
                return
            if self.path == "/":
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/cases":
                self.send(200, json.dumps(records(output / "outbox.sqlite3")).encode())
            elif re.fullmatch(r"/audio/speech-[a-f0-9]{64}\.mp3", self.path):
                path = output / self.path.rsplit("/", 1)[1]
                if path.is_symlink() or not path.is_file():
                    self.send(404, b"{}")
                else:
                    self.send(200, path.read_bytes(), "audio/mpeg")
            else:
                self.send(404, b"{}")

        def do_POST(self):
            if not self.authorized():
                self.send(401, b'{"error":"Authentication required"}')
                return
            if self.path != "/api/cases":
                self.send(404, b"{}")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if (
                    not 0 < size <= 1_000_000
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError("Invalid request size or media type")
                body = json.loads(self.rfile.read(size))
                providers = body["providers"]
                if (
                    not isinstance(providers, list)
                    or not providers
                    or any(p not in PROVIDERS for p in providers)
                ):
                    raise ValueError("Select supported providers")
                case = validate(body["case"])
                outbox = Outbox(output / "outbox.sqlite3")
                try:
                    identifier = outbox.submit(case, providers)
                finally:
                    outbox.close()
                self.send(
                    202,
                    json.dumps(
                        {"case_id": identifier, "status": "queued", "provider_calls": False}
                    ).encode(),
                )
            except (ValueError, KeyError, TypeError):
                self.send(400, b'{"error":"Invalid inspection case or provider list"}')

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/integrations"))
    parser.add_argument("--port", type=int, default=18110)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        handler(
            args.output, required("BRICKCELL_REPORT_USER"), required("BRICKCELL_REPORT_PASSWORD")
        ),
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
