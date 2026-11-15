"""HTTP Server & REST API handler for the SO-101 Architecture Workbench in Physical AI Studio."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .engine import SO101StudioEngine

_UI_EXT_PATH = (
    Path(__file__).resolve().parents[3]
    / "physical-ai-studio"
    / "application"
    / "backend"
    / "src"
    / "so101_architecture"
    / "ui_extension.js"
)


def handle_studio_get(
    engine: SO101StudioEngine, path: str, query_string: str = ""
) -> tuple[int, str, bytes, dict[str, str]] | None:
    """Handle GET requests for `/api/so101/*` and `/api/studio/*` routes.

    Returns `(status_code, content_type, body_bytes, extra_headers)` or `None` if not a studio route.
    """
    qs = parse_qs(query_string)
    project_id = qs.get("project_id", [None])[0]

    if path in ("/api/so101/state", "/api/studio/state"):
        payload = json.dumps(engine.full_studio_snapshot(project_id=project_id)).encode("utf-8")
        return 200, "application/json", payload, {"Cache-Control": "no-store"}

    if path in ("/api/so101/export", "/api/studio/export"):
        payload = json.dumps(engine.export_architecture_bundle(), indent=2).encode("utf-8")
        return 200, "application/json", payload, {"Cache-Control": "no-store"}

    if path in ("/api/so101/ui-extension.js", "/api/studio/ui-extension.js"):
        js = (
            _UI_EXT_PATH.read_bytes() if _UI_EXT_PATH.is_file() else b"// ui_extension.js missing\n"
        )
        return 200, "application/javascript", js, {"Cache-Control": "no-store"}

    for prefix in ("/api/so101/camera/", "/frame/studio/"):
        if path.startswith(prefix) and path.endswith(".jpg"):
            slot = path[len(prefix) : -4]
            jpeg_bytes, src_tag = engine.render_camera_jpeg(slot)
            return (
                200,
                "image/jpeg",
                jpeg_bytes,
                {"Cache-Control": "no-store, max-age=0", "X-Camera-Source": src_tag},
            )

    return None


def handle_studio_post(
    engine: SO101StudioEngine, path: str, body: dict[str, Any]
) -> tuple[int, dict[str, Any]] | None:
    """Handle POST requests for `/api/so101/*` and `/api/studio/*` routes.

    Returns `(status_code, response_dict)` or `None` if not a studio route.
    """
    if path in ("/api/so101/sync-project", "/api/studio/sync-project"):
        custom_cal = body.get("custom_calibration")
        arm = str(body.get("arm", "follower"))
        if isinstance(custom_cal, dict):
            try:
                engine.apply_custom_calibration(arm, custom_cal)
            except ValueError as exc:
                return 400, {"error": str(exc)}
        return 200, engine.full_studio_snapshot(project_id=body.get("project_id"))

    if path in ("/api/so101/kinematics", "/api/studio/kinematics"):
        arm = str(body.get("arm", "follower"))
        cal = engine.follower_cal if arm == "follower" else engine.leader_cal
        try:
            if body.get("q_enc_rad") is not None:
                q_rad = cal.validate_limits(body["q_enc_rad"])
            elif body.get("ticks") is not None:
                q_rad = cal.ticks_to_radians(body["ticks"])
            elif body.get("lerobot_deg") is not None:
                q_rad = cal.lerobot_degrees_to_radians(body["lerobot_deg"])
            elif body.get("normalized") is not None:
                q_rad = engine.normalized_to_radians(body["normalized"], arm=arm)
            else:
                q_rad = engine.virtual_follower_q if arm == "follower" else engine.virtual_leader_q
        except ValueError as exc:
            return 400, {"error": str(exc)}

        if arm == "follower":
            engine.virtual_follower_q = np.asarray(q_rad, dtype=np.float64)
            if body.get("gripper_pct") is not None:
                engine.virtual_follower_gripper_pct = float(body["gripper_pct"])
        else:
            engine.virtual_leader_q = np.asarray(q_rad, dtype=np.float64)
            if body.get("gripper_pct") is not None:
                engine.virtual_leader_gripper_pct = float(body["gripper_pct"])

        if body.get("payload_mass_kg") is not None:
            engine.payload_mass_kg = max(0.0, float(body["payload_mass_kg"]))

        tcp_off = body.get("tcp_offset")
        kin = engine.compute_kinematics_and_jacobian(q_rad, arm=arm, tcp_offset=tcp_off)
        kin["normalized"] = [
            round(float(v), 2) for v in engine.radians_to_normalized(q_rad, arm=arm)
        ]
        dyn = engine.compute_dynamics(
            q_rad, arm=arm, payload_mass_kg=engine.payload_mass_kg, tcp_offset=tcp_off
        )
        return 200, {
            "arm": arm,
            "kinematics": kin,
            "dynamics": dyn,
            "arms": engine.poll_arms_state(),
        }

    if path in ("/api/so101/ik", "/api/studio/ik"):
        try:
            sol = engine.solve_5dof_ik(
                target_xyz_world=body.get("target_xyz_world", [0.22, 0.0, 0.08]),
                pitch_down_deg=float(body.get("pitch_down_deg", 90.0)),
                jaw_yaw_deg=float(body.get("jaw_yaw_deg", 0.0)),
                q_seed=body.get("q_seed"),
            )
        except ValueError as exc:
            return 400, {"error": str(exc)}

        q_rad = sol["q_enc_rad"]
        sol["normalized"] = [
            round(float(v), 2) for v in engine.radians_to_normalized(q_rad, arm="follower")
        ]
        if body.get("update_virtual_arm", True):
            engine.virtual_follower_q = np.asarray(q_rad, dtype=np.float64)
        wp_key = body.get("apply_to_waypoint")
        if wp_key and wp_key in engine.waypoints:
            engine.waypoints[wp_key]["q_enc_rad"] = [round(float(v), 5) for v in q_rad]
            sol["updated_waypoint"] = wp_key
            sol["trajectory"] = engine.build_trajectory_preview(steps_per_segment=10)
        sol["dynamics"] = engine.compute_dynamics(q_rad, arm="follower")
        return 200, sol

    if path in ("/api/so101/teach", "/api/studio/teach"):
        wp_key = str(body.get("waypoint", ""))
        if wp_key not in engine.waypoints:
            return 400, {"error": f"Unknown waypoint: {wp_key!r}"}
        src_arm = str(body.get("source_arm", "leader"))
        grip_in = body.get("gripper_pct")

        if src_arm == "leader":
            if body.get("normalized") is not None:
                q_lead = engine.normalized_to_radians(body["normalized"], arm="leader")
                mapped = engine.map_leader_to_follower(
                    leader_q_enc_rad=q_lead, leader_gripper_pct=grip_in
                )
            elif body.get("ticks") is not None:
                mapped = engine.map_leader_to_follower(
                    leader_ticks=body["ticks"], leader_gripper_pct=grip_in
                )
            elif body.get("q_enc_rad") is not None:
                mapped = engine.map_leader_to_follower(
                    leader_q_enc_rad=body["q_enc_rad"], leader_gripper_pct=grip_in
                )
            else:
                mapped = engine.poll_arms_state()["leader_to_follower_mapping"]
            q_fol = mapped["follower_q_enc_rad"]
            grip_pct = mapped["follower_gripper_pct"] if grip_in is None else float(grip_in)
        else:
            if body.get("normalized") is not None:
                q_fol = [
                    round(float(v), 5)
                    for v in engine.normalized_to_radians(body["normalized"], arm="follower")
                ]
            elif body.get("ticks") is not None:
                q_fol = [
                    round(float(v), 5) for v in engine.follower_cal.ticks_to_radians(body["ticks"])
                ]
            elif body.get("q_enc_rad") is not None:
                q_fol = [round(float(v), 5) for v in body["q_enc_rad"]]
            else:
                q_fol = [round(float(v), 5) for v in engine.virtual_follower_q]
            grip_pct = float(engine.virtual_follower_gripper_pct if grip_in is None else grip_in)

        engine.waypoints[wp_key]["q_enc_rad"] = q_fol
        engine.waypoints[wp_key]["gripper_pct"] = round(grip_pct, 2)
        engine.virtual_follower_q = np.asarray(q_fol, dtype=np.float64)
        engine.virtual_follower_gripper_pct = grip_pct
        return 200, {
            "updated_waypoint": wp_key,
            "waypoint_data": engine.waypoints[wp_key],
            "waypoints": engine.waypoints,
            "kinematics": engine.compute_kinematics_and_jacobian(q_fol, arm="follower"),
            "dynamics": engine.compute_dynamics(q_fol, arm="follower"),
            "trajectory": engine.build_trajectory_preview(steps_per_segment=10),
        }

    if path in ("/api/so101/align", "/api/studio/align"):
        try:
            res = engine.derive_alignment_from_reference_ticks(
                arm=str(body.get("arm", "follower")),
                measured_ticks=body.get("measured_ticks", [2048, 2048, 2048, 2048, 2048]),
                cad_target_deg=body.get("cad_target_deg", [0.0, 93.1, -89.5, 1.5, 90.0]),
                cad_signs=body.get("cad_signs", [1, 1, 1, 1, 1]),
            )
            return 200, res
        except ValueError as exc:
            return 400, {"error": str(exc)}

    if path in ("/api/so101/bowls", "/api/studio/bowls"):
        for k in ("source", "good", "defective"):
            val = body.get(k)
            if isinstance(val, list) and len(val) == 3:
                engine.bowls[k]["pos_world_m"] = [round(float(v), 4) for v in val]
        engine._refresh_ik_waypoints()
        return 200, {
            "bowls": engine.bowls,
            "waypoints": engine.waypoints,
            "trajectory": engine.build_trajectory_preview(steps_per_segment=10),
        }

    return None


def render_standalone_studio_html() -> bytes:
    """Return a minimal HTML shell that loads `/api/so101/ui-extension.js` and opens the workbench."""
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Physical AI Studio · SO-101 Model-Based Pick & Place Architecture</title>
  <script defer src="/api/so101/ui-extension.js"></script>
</head>
<body style="margin:0;background:#111318;color:#e2e8f0;font-family:Inter,system-ui,sans-serif;">
  <div aria-label="Header navigation" style="height:56px;background:#181b22;border-bottom:1px solid #2a303c;display:flex;align-items:center;padding:0 20px;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:14px;" role="tablist">
      <strong style="color:#f8fafc;font-size:15px;">Physical AI Studio</strong>
      <span style="color:#94a3b8;font-size:13px;">SO-101 Model-Based Pick &amp; Place Edition</span>
    </div>
  </div>
  <script>
    window.location.hash = "#so101-architecture";
  </script>
</body>
</html>
"""
    return html.encode("utf-8")


class StudioHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], engine: SO101StudioEngine) -> None:
        super().__init__(server_address, StudioHTTPHandler)
        self.engine = engine


class StudioHTTPHandler(BaseHTTPRequestHandler):
    server: StudioHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html", "/studio"):
            body = render_standalone_studio_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        res = handle_studio_get(self.server.engine, parsed.path, parsed.query)
        if res is not None:
            code, ctype, data, headers = res
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        res = handle_studio_post(self.server.engine, parsed.path, body)
        if res is not None:
            code, payload = res
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Physical AI Studio — SO-101 Model-Based Pick-and-Place Architecture Workbench"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--armd-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--export-bundle", type=Path, help="Export standalone architecture bundle JSON and exit"
    )
    args = parser.parse_args(argv)

    engine = SO101StudioEngine(armd_url=args.armd_url)
    if args.export_bundle is not None:
        bundle = engine.export_architecture_bundle()
        args.export_bundle.parent.mkdir(parents=True, exist_ok=True)
        args.export_bundle.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        print(f"Exported SO-101 Architecture Bundle to {args.export_bundle}")
        return 0

    srv = StudioHTTPServer((args.host, args.port), engine)
    print(
        f"Physical AI Studio (SO-101 Architecture Workbench) listening on http://{args.host}:{args.port}"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
