# Saved review pages and live viewers

The HTML snapshots in `scripts/review_pages/` preserve the requested local
review pages as they appeared on 14 November 2026. They contain historical
observations, instructions and result summaries, not new live predictions.
Restoring them overwrites only these three HTML files; source captures, reports,
point clouds and model files remain untouched.

```bash
uv sync --locked
uv run python scripts/restore_review_pages.py
uv run python -m http.server 18096 --bind 127.0.0.1 --directory artifacts
```

| Local URL | Versioned source |
|---|---|
| `http://127.0.0.1:18096/brick-001-reconstruction-audit/review.html` | `scripts/review_pages/brick-001-reconstruction-audit/review.html` |
| `http://127.0.0.1:18096/brick-001-composite-review/review.html` | `scripts/review_pages/brick-001-composite-review/review.html` |
| `http://127.0.0.1:18096/brick-002-orientation-check/orientation.html` | `scripts/review_pages/brick-002-orientation-check/orientation.html` |
| `http://127.0.0.1:18094/` | `scripts/live_depth.py` (Intel SDK service, forwarded from port 8094) |
| `http://127.0.0.1:18097/` | `scripts/patchcore_depth_viewer.py` with `configs/rgbd-viewer.json` |

`?updated=actual-brick`, `?result=line` and `?training=patchcore` refer to the same
orientation HTML. They do not select separate stored versions or rerun training.

The static pages depend on local files intentionally excluded from Git. A fresh
checkout restores the HTML, but images, plots, PLY clouds and linked JSON reports
will be unavailable until the corresponding local artifacts are supplied or
regenerated. The orientation page references `brick-002-full-baseline-02/`,
`brick-002-full-raised-01/`, `brick-002-line-detection/`,
`brick-002-top-change-comparison/` and `pilots/brick-002-line-v1/`. The composite
page needs its candidate PLY/PNG plus the four pose-review photo folders. The
audit page needs its generated audit reports, clouds and plots. Keep the original
relative directory layout under `artifacts/`.

Relevant generators are `scripts/reconstruction_audit.py`,
`scripts/align_pose_preview.py`, `scripts/capture_rgbd_sequence.py`,
`scripts/compare_top_depth.py`, `scripts/detect_brick_line.py` and
`scripts/train_line_pilot.py`. Consult `docs/RECONSTRUCTION.md` for reconstruction
inputs and `docs/RGBD_VIEWER.md` for sensor contracts and depth preparation.
Rerunning reconstruction audit writes its generated review template; rerun the
restore command afterward to recover this saved HTML snapshot if needed.

The live services require separate startup and camera access. Restoring static
HTML does not start them, create SSH tunnels, change hardware or make loopback
URLs accessible on GitHub. Neither the saved fused cloud nor its live visualization
is connected to automatic point-cloud defect detection.
