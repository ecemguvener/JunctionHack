"""Loopback live D415 point-cloud review; run in a separate Intel SDK environment."""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><meta charset=utf-8><title>Live D415 depth</title>
<style>body{background:#111;color:white;font:16px system-ui}section{display:inline-block;width:48%}canvas,img{width:100%}canvas{background:#18222a}button{font-size:18px}</style>
<h1>Live 3D camera views</h1><p>Separate camera coordinates. Drag to rotate. Scroll to zoom. Depth range: 0.15–1.5 m. These views are not merged.</p>
<div id=views></div><script>
for(const name of ['top','plate']){
let section=document.createElement('section');section.innerHTML=`<h2>${name}</h2><p class=status>Connecting…</p><img><canvas width=640 height=480></canvas>`;views.append(section);
let canvas=section.querySelector('canvas'),ctx=canvas.getContext('2d'),yaw=0,pitch=0,zoom=500,drag=false,last;
canvas.onpointerdown=e=>{drag=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointerup=()=>drag=false;canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-last[0])*.01;pitch+=(e.clientY-last[1])*.01;last=[e.clientX,e.clientY]}};
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(100,Math.min(3000,zoom*Math.exp(-e.deltaY*.001)))};
async function tick(){try{let r=await fetch('/points/'+name);let d=await r.json();if(!r.ok)throw Error(d.error);section.querySelector('.status').textContent=`${d.points.length} points · age ${d.age_s.toFixed(2)} s`;
section.querySelector('img').src='/rgb/'+name+'?t='+Date.now();ctx.clearRect(0,0,640,480);
for(let p of d.points){let x=p[0],y=p[1],z=p[2]-.6;let a=x*Math.cos(yaw)+z*Math.sin(yaw),b=-x*Math.sin(yaw)+z*Math.cos(yaw);let v=y*Math.cos(pitch)-b*Math.sin(pitch);ctx.fillStyle=`rgb(${p[3]},${p[4]},${p[5]})`;ctx.fillRect(320+a*zoom,240+v*zoom,2,2)}}catch(e){section.querySelector('.status').textContent=e.message}setTimeout(tick,300)}tick();}
</script>"""


def main():
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    packets, lock, stop = {}, threading.Lock(), threading.Event()

    def worker(name, serial):
        pipeline = rs.pipeline()
        started = False
        try:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            profile = pipeline.start(config)
            started = True
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)
            pc = rs.pointcloud()
            while not stop.is_set():
                frames = align.process(pipeline.wait_for_frames(3000))
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                rgb = np.asanyarray(color.get_data())
                xyz = (
                    np.asanyarray(pc.calculate(depth).get_vertices())
                    .view(np.float32)
                    .reshape(480, 640, 3)[::4, ::4]
                )
                colors = rgb[::4, ::4, ::-1]
                valid = np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0.15) & (xyz[:, :, 2] < 1.5)
                points = np.column_stack((xyz[valid], colors[valid])).round(4).tolist()
                ok, jpeg = cv2.imencode(".jpg", rgb)
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                intr = color.profile.as_video_stream_profile().get_intrinsics()
                packet = dict(
                    points=points,
                    time=time.time(),
                    jpeg=jpeg.tobytes(),
                    serial=serial,
                    depth_scale_m=scale,
                    intrinsics=dict(
                        fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy, coeffs=intr.coeffs
                    ),
                )
                with lock:
                    packets[name] = packet
        except Exception as exc:
            with lock:
                packets[name] = {"error": str(exc)}
        finally:
            if started:
                pipeline.stop()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            route = self.path.split("?")[0]
            status, kind, body = 200, "text/html", PAGE.encode()
            if route != "/":
                parts = route.strip("/").split("/")
                if (
                    len(parts) != 2
                    or parts[0] not in ("points", "rgb")
                    or parts[1] not in ("top", "plate")
                ):
                    self.send_error(404)
                    return
                with lock:
                    p = dict(packets.get(parts[1], {"error": "Waiting for camera"}))
                age = time.time() - p.get("time", 0)
                if p.get("error") or age > 3:
                    status, kind = 503, "application/json"
                    body = json.dumps({"error": p.get("error", "Stale camera frame")}).encode()
                elif parts[0] == "rgb":
                    kind, body = "image/jpeg", p["jpeg"]
                else:
                    p.pop("jpeg")
                    p["age_s"] = age
                    kind, body = "application/json", json.dumps(p, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    threads = [
        threading.Thread(target=worker, args=(name, serial), daemon=True)
        for name, serial in [("top", "350122060653"), ("plate", "243622060450")]
    ]
    for thread in threads:
        thread.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()
        for thread in threads:
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
