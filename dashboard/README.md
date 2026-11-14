# Station twin dashboard

One page for the judges. It shows how the station sees the world:

- **Left: Top camera.** Switch between the **MuJoCo sim** render and the **real** D415 Top
  camera with the tabs or the `T` key. The two images use the same measured camera pose and
  intrinsics, so you can flick between them.
- **Right: Depth point cloud.** The fused points of both D415 cameras in the robot base frame.
  It opens from the Top camera's point of view. Click full screen (⤢) to move freely around
  the plates with Blender-style controls.
- **Right column: overlays** on the sim/real view: *Defects* (anomaly boxes, all visible or
  only on the focus block) and *Focus block* (the block the arm selected to pick and inspect).

The dashboard only reads. It sends no robot or simulator commands and never opens a camera.

## Run it

It runs on the Intel laptop as the user service `twin-dashboard` on `127.0.0.1:8096`.

```bash
dashboard/deploy.sh            # copy to intel:~/brickcell-dashboard and restart the service
dashboard/deploy.sh --dry-run  # show what would change
```

- On the Intel laptop: open `http://127.0.0.1:8096/`.
- From another computer: `ssh -N -L 8096:127.0.0.1:8096 intel`, then open
  `http://127.0.0.1:8096/`.

The service uses `uv run --offline` with numpy and OpenCV (both already in the uv cache on
Intel). Logs: `journalctl --user -u twin-dashboard -f`.

## Sources

| Data | From | Notes |
|---|---|---|
| Real Top frames | live depth service `:8094/rgb/top` | re-encoded to JPEG quality 75 |
| Sim Top frames | MuJoCo sim server `:18081/camera/overhead.png` | 4 fps; the sim is **not twin-synced** |
| Camera poses + intrinsics | sim server `:18081/scene` | the one source of truth for both views |
| Point clouds | live depth service `:8094/points/top`, `/points/plate` | every 4th pixel, 0.15–1.5 m |
| Sim state | sim server `:18081/health` | shown as IDLE / RUNNING |

Start the live depth service and the sim server first. If a source stops, its view shows
**No signal** with the reason, and the header shows **Degraded**.

### Table levelling

The sim scene's camera poses can be out of date when a camera moves. Every 10 s the
dashboard fits the table plane (RANSAC) in each camera's cloud and moves that cloud so the
table is at z = 0 (at most 8° and 8 cm). `/api/status` reports the correction per camera
(`table.tilt_deg`, `table.offset_mm`). This corrects height and tilt only, not a horizontal
shift or a yaw. A large correction means that the camera needs a new calibration.

## Keys

| Key | Action |
|---|---|
| `T` | Sim / real |
| `D` | Defects overlay |
| `F` | Focus block overlay |
| `Z` | Camera zoom 1× / 2× |
| `Esc` | Leave full screen |
| `0` `7` `1` `3` (`Ctrl` = opposite) | Depth full screen: camera, top, front, right view |
| `5` | Depth full screen: perspective / orthographic |
| `8` / `9` | Show or hide the Top / Plate D415 cloud and its frustum (also the buttons in the depth bar) |
| `Home` or `.` | Depth full screen: centre on the plates |

Depth full screen mouse: drag = orbit, Shift + drag or right drag = pan, wheel or Ctrl + drag
= zoom. The axis gizmo and the buttons at the bottom left do the same.

## API

| Request | Result |
|---|---|
| `GET /api/status` | Health of every source, overlay state |
| `GET /api/config` | Top/Plate camera poses and intrinsics, pivot, workspace |
| `GET /frame/{real,sim}.jpg?after=SEQ` | Next frame after `SEQ` (waits up to 2 s, else 204) |
| `GET /stream/{real,sim}.mjpg` | MJPEG stream (for simple viewers; the page does not use it) |
| `GET /api/cloud?after=SEQ` | Next fused cloud (gzip if accepted). Format: `cloud_format` in `/api/config` |
| `POST /api/anomalies` | **Stub input** for a defect detector (below) |
| `POST /api/focus` | **Stub input** for the pick planner (below) |

## Stubs: not integrated yet

The anomaly detector and the pick planner POST JSON to the
dashboard. Boxes are pixels in the 640×480 Top camera image (the same for sim and real).
A publisher connection expires after 5 s (`--overlay-stale`). Inspection evidence
expires independently using acquisition time. Reposting a result cannot renew it.
The following legacy payloads remain accepted, but display **Unverified result**
and hide their boxes because capture provenance is missing.

```bash
curl -X POST http://127.0.0.1:8096/api/anomalies -H 'Content-Type: application/json' \
  -d '{"source": "patchcore-top", "items": [{"box": [238, 171, 250, 183], "score": 0.83,
       "label": "dark mark", "object_id": "part-002"}]}'

curl -X POST http://127.0.0.1:8096/api/focus -H 'Content-Type: application/json' \
  -d '{"source": "planner", "object_id": "part-002", "label": "2x4 red", "box": [222, 161, 270, 196]}'
```

`score` is a raw anomaly distance, not a probability. Send `"items": []` or
`"object_id": null` to clear.

## Tests

```bash
uv run --no-project --python 3.12 --with numpy --with opencv-python-headless --with pytest \
  pytest dashboard/tests
```

The tests use fake upstream services: no hardware, no sim.

## Capture evidence: publisher upgrade required

The deployed controller was observed resending cached payloads as a keepalive.
This version separates `connected` (transport) from `drawable` (eligible object
estimate). Existing camera/depth feeds continue working. Legacy result publishers
do not need an immediate code change to keep posting, but their boxes are hidden.
Coordinate rollout with the controller owner; this change is not deployed automatically.

Include an `evidence` object in both anomaly and focus POST bodies:

```python
payload["evidence"] = {
    "frame_id": snapshot.frame_id,  # actual source observation; retain on heartbeat
    "camera_id": "plate",  # the camera whose image was scored
    "captured_at": snapshot.captured_at,  # Unix seconds at acquisition, NOT send time
    "mode": "real",  # "sim" for synthetic observations
    "box_space": "top_projected_object",
    "method": "dark_line_heuristic",  # identify the actual algorithm
    "model_version": "heuristic-v1",
    "threshold_version": None,  # no invented calibration/version
}
```

`snapshot` above illustrates the contract; map to your recorder's actual fields.
Do not stamp old results with the current time. Reusing a remembered frame ID with
a changed acquisition time is rejected. The server retains up to 1,024 frame keys
per endpoint; this is a debugging guard, not authentication of publisher claims.
Clock skew beyond 0.5 s in the future is rejected; align clocks at the station.

`top_projected_object` means an approximate whole-object box projected into the
Top image. The UI explicitly labels it **OBJECT ESTIMATE**, not defect localization.
It does not prove the camera-to-robot transform or object identity is correct.
Results tagged real/sim never overlay the other view. A `source_image_pixels`
result is retained but hidden on the live view: exact defect regions need the
corresponding scored frame, which the existing feed contract does not provide.

Method/model/threshold metadata is shown in the sidebar. Unknown calibration
stays unknown. Empty detections are not labelled GOOD. Publisher heartbeat age
and capture age are distinct. This dashboard remains read-only and does not
authorize sorting or change the robot controller's decisions.

Additional browser-rule tests (Node.js):

```bash
node --test dashboard/tests/test_evidence.cjs
node --check dashboard/static/app.js
```
