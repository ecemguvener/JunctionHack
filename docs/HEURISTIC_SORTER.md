# Opt-in dark-mark sorter

`deploy-cell --inspection-mode lab-dark-mark` selects the CIE-Lab contrast rule
used by `origin/demo_1`, without requiring trained PatchCore exports. The default
remains `patchcore`, requiring calibrated real plate and wrist models.

In the explicit Lab mode, a valid plate-camera view with dark-mark score at or
above `--dark-mark-threshold` (default **16**) flags the held brick for the pink
plate. A complete scan without a flag selects green. Scores are Lab contrast
distances, not probabilities; 16 is an operator-selected heuristic setting, not
a newly calibrated defect threshold or a claim of real defect accuracy.

The restored path requires **all seven scheduled plate views** to be valid and
have distinct frame IDs. This is stricter than the old branch's three-view
minimum. Invalid/clipped/empty views, repeated frames, inspector failures and
timeouts block sorting. Grasp verification, unscanned-sort refusal, motion
limits and emergency hold remain in place. A timed-out worker is retired.
Wrist imagery still supports the existing grasp/visualization workflow, but it
does not supply heuristic classification or plate scan coverage in this mode.
The mode cannot silently mix Lab scores with loaded PatchCore models.

After an operator verifies a clear workspace, an empty gripper, and no other
active physical sorter, run one cycle from an updated Intel checkout:

```sh
PYTHONPATH=src .venv/bin/python -m brickcell.block_inspector.cli deploy-cell \
  --inspection-mode lab-dark-mark --dark-mark-threshold 16 \
  --max-cycles 1 --hud-port 0 \
  --output-dir artifacts/heuristic-sort-run
```

This command issues physical motion. It does not recover or resume a brick
already held by an interrupted cycle. Do not start another controller if one is
already running. `--hud-port 0` avoids taking the existing camera-wall/HUD port.

An isolated source snapshot is staged at:

`/home/intel/brickcell-deploy/artifacts/heuristic-sorter-v1`

It uses the existing Intel environment and references the existing assets/tools
through symlinks, leaving the teammate's deployment source unchanged. To select
that snapshot, use this one-cycle command from `/home/intel/brickcell-deploy`:

```sh
PYTHONPATH=/home/intel/brickcell-deploy/artifacts/heuristic-sorter-v1/src \
  .venv/bin/python -m brickcell.block_inspector.cli deploy-cell \
  --inspection-mode lab-dark-mark --dark-mark-threshold 16 \
  --max-cycles 1 --hud-port 0 \
  --output-dir artifacts/heuristic-sort-run
```

Startup/import and CLI help have been checked on Intel. No physical cycle has
been launched with this snapshot. Reports identify `lab_dark_mark_plate_sort`
and the selected contrast rule rather than claiming calibrated PatchCore use.
The existing source function excludes plate/gripper colors and uses narrow
stroke geometry; it has not been validated for every brick color, defect width,
surface, lighting condition or hidden face. This is the legacy operating rule,
not an established all-defect detector.
