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
import math
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
    """The newest per-camera malform results. Nothing sets a camera until its model connects."""

    def __init__(self, stale_s: float):
        self.lock = threading.Lock()
        self.stale_s = stale_s
        self.malformed: bool | None = None
        self.detail = ""
        self.stamp: float | None = None
        self.camera = "side"
        self.frame_id = None
        self.source = "unknown"
        self._by_camera: dict[str, dict] = {}

    def update(self, payload: dict) -> None:
        malformed = payload.get("malformed")
        if malformed is not None and type(malformed) is not bool:
            raise ValueError('"malformed" must be true, false, or null')
        detail = payload.get("detail", "")
        if not isinstance(detail, str) or len(detail) > 200:
            raise ValueError('"detail" must be a string of 200 characters or less')
        now = time.time()
        captured_at = payload.get("captured_at", now)
        if type(captured_at) not in (int, float) or not math.isfinite(captured_at):
            raise ValueError("Invalid captured_at")
        if not 0 <= now - captured_at <= self.stale_s:
            raise ValueError("Detection frame is stale or future-dated")
        camera = payload.get("camera", "side")
        if camera not in ("side", "overhead", "wrist"):
            raise ValueError("Unsupported camera")
        frame_id = payload.get("frame_id")
        if frame_id is not None and (not isinstance(frame_id, str) or len(frame_id) > 200):
            raise ValueError("Invalid frame_id")
        source = payload.get("source", "unknown")
        if not isinstance(source, str) or len(source) > 200:
            raise ValueError("Invalid detection source")
        with self.lock:
            self._by_camera[camera] = {
                "malformed": malformed,
                "detail": detail,
                "stamp": None if malformed is None else float(captured_at),
                "camera": camera,
                "frame_id": frame_id,
                "source": source,
            }
            self.malformed, self.detail = malformed, detail
            self.stamp = None if malformed is None else float(captured_at)
            self.camera, self.frame_id, self.source = camera, frame_id, source

    def _entry_status(self, camera: str, entry: dict, now: float) -> dict:
        stamp = entry["stamp"]
        malformed = entry["malformed"]
        age = now - stamp if stamp is not None else None
        if malformed is None or age is None:
            state = "none"
        elif age > self.stale_s:
            state = "stale"
        else:
            state = "malformed" if malformed else "ok"
        return {
            "state": state,
            "detail": entry["detail"],
            "age_s": age,
            "camera": camera,
            "frame_id": entry["frame_id"],
            "source": entry["source"],
        }

    def status(self) -> dict:
        now = time.time()
        with self.lock:
            if not self._by_camera:
                return {
                    "state": "none",
                    "detail": self.detail,
                    "age_s": None,
                    "camera": self.camera,
                    "frame_id": self.frame_id,
                    "source": self.source,
                    "cameras": {},
                }
            per_camera = {
                cam: self._entry_status(cam, entry, now) for cam, entry in self._by_camera.items()
            }
        malformed_list = [s for s in per_camera.values() if s["state"] == "malformed"]
        ok_list = [s for s in per_camera.values() if s["state"] == "ok"]
        stale_list = [s for s in per_camera.values() if s["state"] == "stale"]
        if malformed_list:
            primary = min(
                malformed_list, key=lambda s: s["age_s"] if s["age_s"] is not None else 0.0
            )
            active = malformed_list + ok_list
        elif ok_list:
            primary = min(ok_list, key=lambda s: s["age_s"] if s["age_s"] is not None else 0.0)
            active = ok_list
        elif stale_list:
            primary = min(stale_list, key=lambda s: s["age_s"] if s["age_s"] is not None else 0.0)
            active = [primary]
        else:
            primary = next(iter(per_camera.values()))
            active = [primary]
        if len(active) > 1:
            combined = " | ".join(
                f"{s['camera']}: {s['detail']}" if s["detail"] else f"{s['camera']}: {s['state']}"
                for s in active
            )
            detail = combined[:200]
        else:
            detail = primary["detail"]
        return {
            "state": primary["state"],
            "detail": detail,
            "age_s": primary["age_s"],
            "camera": primary["camera"],
            "frame_id": primary["frame_id"],
            "source": primary["source"],
            "cameras": per_camera,
        }


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
