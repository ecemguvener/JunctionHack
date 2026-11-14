"""Camera wall: three camera feeds side by side, with a malform status panel.

Standard library only. The server reads JPEG frames from HTTP sources (by default
the three-camera bench on 127.0.0.1:8092). The page gets each new frame from
GET /frame/<camera>?after=<sequence number>.

Malform detection is not part of this program. A model sets the result with:

    POST /api/detection  {"malformed": true | false | null, "detail": "optional text"}

The page shows "No model result" until a result comes, and when a result is old.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
DEFAULT_CAMERAS = [
    "overhead=Top=http://127.0.0.1:8092/image/overhead",
    "side=Plate=http://127.0.0.1:8092/image/side",
    "wrist=Gripper=http://127.0.0.1:8092/image/wrist",
]
STALE_FRAME_S = 2.0
MAX_BODY = 8192


class Feed:
    """Read the newest JPEG from one HTTP source in a background thread."""

    def __init__(self, name: str, label: str, url: str, fps: float):
        self.name, self.label, self.url = name, label, url
        self.period = 1.0 / fps
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.seq = 0
        self.stamp = 0.0
        self.error = "Waiting for camera"
        self.stopped = threading.Event()
        threading.Thread(target=self._loop, name=f"feed-{name}", daemon=True).start()

    def _loop(self) -> None:
        while not self.stopped.is_set():
            start = time.monotonic()
            try:
                with urlopen(self.url, timeout=3) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}")
                    body = response.read()
                if not body.startswith(b"\xff\xd8"):
                    raise RuntimeError("Source did not send a JPEG")
                with self.cond:
                    if body != self.jpeg:
                        self.jpeg, self.seq = body, self.seq + 1
                    self.stamp, self.error = time.time(), ""
                    self.cond.notify_all()
            except Exception as exc:  # noqa: BLE001 - any source failure is shown to the user
                with self.cond:
                    self.error = str(exc) or type(exc).__name__
                self.stopped.wait(1.0)
            self.stopped.wait(max(0.0, self.period - (time.monotonic() - start)))

    def wait_next(self, seq: int, timeout: float) -> tuple[int, bytes | None]:
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq or self.stopped.is_set(), timeout)
            return self.seq, self.jpeg

    def status(self) -> dict:
        with self.cond:
            age = time.time() - self.stamp if self.stamp else None
            live = self.jpeg is not None and age is not None and age < STALE_FRAME_S
            error = self.error or ("" if live else "No new frame")
            return {"name": self.name, "label": self.label, "live": live, "error": error}


class Detection:
    """The newest malform result. Nothing sets it until a model is connected."""

    def __init__(self, stale_s: float):
        self.lock = threading.Lock()
        self.stale_s = stale_s
        self.malformed: bool | None = None
        self.detail = ""
        self.stamp: float | None = None

    def update(self, payload: dict) -> None:
        malformed = payload.get("malformed")
        if malformed not in (True, False, None):
            raise ValueError('"malformed" must be true, false, or null')
        detail = payload.get("detail", "")
        if not isinstance(detail, str) or len(detail) > 200:
            raise ValueError('"detail" must be a string of 200 characters or less')
        with self.lock:
            self.malformed, self.detail = malformed, detail
            self.stamp = None if malformed is None else time.time()

    def status(self) -> dict:
        with self.lock:
            age = time.time() - self.stamp if self.stamp is not None else None
            if self.malformed is None or age is None:
                state = "none"
            elif age > self.stale_s:
                state = "stale"
            else:
                state = "malformed" if self.malformed else "ok"
            return {"state": state, "detail": self.detail, "age_s": age}


def make_handler(feeds: dict[str, Feed], detection: Detection):
    page = (HERE / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def send(self, status: int, kind: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, data: dict) -> None:
            self.send(status, "application/json", json.dumps(data).encode())

        def do_GET(self):
            route, _, query = self.path.partition("?")
            if route == "/":
                self.send(200, "text/html; charset=utf-8", page)
            elif route == "/api/status":
                self.send_json(
                    200,
                    {
                        "cameras": [f.status() for f in feeds.values()],
                        "detection": detection.status(),
                    },
                )
            elif route == "/api/detection":
                self.send_json(200, detection.status())
            elif route.startswith("/frame/") and route[7:] in feeds:
                self.frame(feeds[route[7:]], query)
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/detection":
                self.send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Send a JSON object")
                detection.update(payload)
                self.send_json(200, detection.status())
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})

        def frame(self, feed: Feed, query: str) -> None:
            """Send the newest JPEG. With ?after=N, wait up to 2 s for a frame newer than N."""
            after = None
            for part in query.split("&"):
                if part.startswith("after="):
                    try:
                        after = int(part[6:])
                    except ValueError:
                        pass
            if after is None:
                seq, jpeg = feed.wait_next(-1, timeout=0)
            else:
                seq, jpeg = feed.wait_next(after, timeout=2)
            if jpeg is None or not feed.status()["live"]:
                self.send_json(503, {"error": feed.status()["error"]})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Seq", str(seq))
            self.end_headers()
            self.wfile.write(jpeg)

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A browser that closes a connection early is normal. Do not log it.
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def parse_camera(spec: str) -> tuple[str, str, str]:
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(parts) or not parts[0].isidentifier():
        raise argparse.ArgumentTypeError("Use NAME=LABEL=URL, for example side=Side=http://...")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument(
        "--camera",
        action="append",
        type=parse_camera,
        help="NAME=LABEL=URL of a JPEG frame. Use it 3 times. Default: the bench on :8092.",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Highest read rate per camera.")
    parser.add_argument(
        "--stale-after",
        type=float,
        default=5.0,
        help="Seconds after which a model result shows as old.",
    )
    args = parser.parse_args()
    cameras = args.camera or [parse_camera(s) for s in DEFAULT_CAMERAS]
    if len({name for name, _, _ in cameras}) != len(cameras):
        parser.error("Camera names must be different")

    feeds = {name: Feed(name, label, url, args.fps) for name, label, url in cameras}
    detection = Detection(args.stale_after)
    server = Server((args.host, args.port), make_handler(feeds, detection))
    print(f"Camera wall ready on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for feed in feeds.values():
            feed.stopped.set()
            with feed.cond:
                feed.cond.notify_all()
        server.server_close()


if __name__ == "__main__":
    main()
