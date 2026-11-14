# Dashboard design mockups

Five clickable design studies for the station twin dashboard. Design 1 (Graphite) was chosen
and built as the live page in `dashboard/`.

| File | Style |
|---|---|
| `1-graphite.html` | Dark control room, orange accent (chosen) |
| `2-datasheet.html` | Light paper, black rules, numbered sections |
| `3-hmi.html` | Machine console, status lamps, rocker switches |
| `4-instrument.html` | No sidebar, edge rulers, floating options |
| `5-appliance.html` | Warm grey, rounded cards, one orange accent |

Open a file in a browser (no server needed). All data is **mock data**:

- `assets/real-top.jpg`: one real Top D415 frame (2026-11-14 05:09 UTC). A person in the
  top-right corner is pixelated.
- `assets/sim-top.jpg`: MuJoCo twin render of the same scene with hand-placed bricks.
- `assets/data.js`: the real Top + Plate point clouds of that frame, brick boxes and
  invented anomaly boxes and scores.

`assets/engine.js` is the shared behaviour; each HTML file is only markup and CSS.
