# Camera wall

The camera wall shows the three camera feeds. The Top and Plate views are side by side at the top, as large as the screen permits. The Gripper view is smaller and centred below them. A small panel at the bottom centre shows the malform result.

The program runs on the Intel laptop (see [ssh.md](../ssh.md)). It uses the Python standard library only.

## Where the frames come from

The camera wall does not open the cameras. It reads JPEG frames from the three-camera bench (`multicam_server.py`, `127.0.0.1:8092`), which holds the cameras:

| Tile | Source |
|---|---|
| Top (top left) | `http://127.0.0.1:8092/image/overhead` (D415, serial 318123021715) |
| Plate (top right) | `http://127.0.0.1:8092/image/side` (D415, serial 344223025431) |
| Gripper (below, smaller) | `http://127.0.0.1:8092/image/wrist` (Sonix USB camera) |

If the bench stops, the tiles show "No signal". To use other sources, give `--camera NAME=LABEL=URL` three times. The first camera goes top left, the second top right, and the third below.

The bench draws its crop rectangle (yellow) on each frame. The camera wall shows the frames as it gets them.

## Malform result

Malform detection is not part of this program. Until a model sends a result, the panel shows "No model result". To send a result:

```bash
curl -X POST http://127.0.0.1:8093/api/detection \
  -H 'Content-Type: application/json' \
  -d '{"malformed": true, "detail": "Side view"}'
```

| `malformed` | Panel |
|---|---|
| `true` | Red: "Malform detected". All tiles get a red border. |
| `false` | Green: "No malform detected". |
| `null` | Grey: "No model result". |

`detail` is optional text (200 characters or less). If no new result comes in 5 s (`--stale-after`), the panel shows "Model result is old".

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /` | The page. |
| `GET /api/status` | Camera state and malform result. |
| `GET /frame/<camera>?after=<seq>` | The newest JPEG. It waits up to 2 s for a frame newer than `seq`. The header `X-Frame-Seq` gives the number of the frame. |
| `GET /api/detection`, `POST /api/detection` | Read or set the malform result. |

## Deploy

From the repository root, on a computer with SSH access to `intel`:

```bash
camera-ui/deploy.sh             # Copy, restart the service, open the page on the laptop display.
camera-ui/deploy.sh --no-show   # Do not open the page on the laptop display.
camera-ui/deploy.sh --dry-run   # Show what rsync copies. Do not change the server.
```

The script copies `camera-ui/` to `/home/intel/camera-ui/`. Then it installs and restarts the systemd **user** service `camui` (`127.0.0.1:8093`). It does not use `sudo`.

| Task | Command (on the laptop) |
|---|---|
| Show the status | `systemctl --user status camui` |
| Show the log | `journalctl --user -u camui -n 50` |
| Open the page full screen | `~/camera-ui/show.sh` |
| Close the full-screen page | `~/camera-ui/show.sh stop`, or push Alt+F4 |
| Stop the service | `systemctl --user stop camui` |

To see the page from your computer: `ssh -N -L 8093:localhost:8093 intel`, then open `http://localhost:8093/`.

## Run without the laptop

```bash
python3 camera-ui/camui.py --port 8093 \
  --camera a=A=http://host/a.jpg --camera b=B=http://host/b.jpg --camera c=C=http://host/c.jpg
```
