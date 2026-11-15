# Camera-view detection warning

`camui.py` and `index.html` preserve Intel's existing three-camera wall. Fresh
`POST /api/detection` events turn only the indicated camera red, using a tint and
border. This is a recent-result warning indicator, not a pixel-level defect mask.
Stale/offline/unknown results clear the warning. The camera JPEG itself is retained.

The display-only bridge `scripts/publish_camui_detection.py` consumes the existing
live dark-mark publisher's `events.jsonl` on Intel and posts its actual result.
It requires fresh real plate-camera identity, supported heuristic provenance,
finite distance/contrast, and a valid view. Acquisition time is preserved, so
repeated posts cannot extend an old warning. This bridge does not command an arm.

Current source: the existing `publish_mark_inspection.py` heuristic publisher,
which tracks one unambiguous red brick. This is **not a trained PatchCore model**
and does not cover all brick colours. It is separate from the arm's explicit
operator-provided colour routing. Invalid/ambiguous/non-red views show no result,
not a clean verdict. No full-brick accuracy is claimed.

Intel:

```sh
python3 /home/intel/camera-ui/publish_camui_detection.py \
  --events /home/intel/brickcell-deploy/artifacts/camera-wall-runtime-v1/events.jsonl
```

Existing service: `systemctl --user restart camui.service`. Current loopback URL:
`http://127.0.0.1:8093/`. Reload an already-open page to load the red tint styles.
The deployed bridge runs in tmux socket `camui-detection-bridge`, session `bridge`.
