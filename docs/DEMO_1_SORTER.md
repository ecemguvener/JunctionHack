# demo_1-sorter branch

This branch starts at `origin/demo_1` commit
`83dc84782400479ff77196e25ecadc2031d75f31` and merges the tested sorting and
camera-wall implementation from `anomaly` through `0df11b7`, including its
required modules and regression tests.

The functional sorting changes are:

- `271ceee`: restore the Lab dark-mark physical sorting path with complete plate scans.
- `210afc8`: make model-free dark-mark sorting the default CLI/API/launcher policy.
- `0df11b7`: use **PatchCore Inspection** as the camera-wall workflow title,
  while retaining the actual detector type and provenance in backend records.

A normal `deploy-cell` launch uses the plate-camera contrast rule, without
PatchCore model files. A detected qualifying dark mark selects pink. A complete,
valid scan with no detected mark selects green. Missing/invalid/duplicate views,
inspector errors and timeouts block sorting. All seven scheduled plate views
must be valid, and grasp and motion safeguards remain active.

The implementation can issue physical sorting commands, but no successful
physical sort was established while creating this branch. The inspection rule
has not been validated for every defect, brick color, occlusion or lighting
condition. The workflow title does not change which detector is running.

## Get this branch

```sh
git fetch origin
git switch demo_1-sorter
git pull --ff-only origin demo_1-sorter
```

On a PowerShell laptop, synchronize the branch to Intel without starting a cycle:

```powershell
.\scripts\deploy_to_intel.ps1 -SyncOnly
```

After an operator verifies a clear workspace, an empty gripper and no other
active sorter, launch one cycle:

```powershell
.\scripts\deploy_to_intel.ps1 -MaxCycles 1
```

Alternatively, from the updated Intel checkout:

```sh
PYTHONPATH=src .venv/bin/python -m brickcell.block_inspector.cli deploy-cell \
  --max-cycles 1 --hud-port 0 \
  --output-dir artifacts/heuristic-sort-run
```

These launch commands move the physical arm. They do not resume an already-held
brick after an interrupted cycle. `--hud-port 0` avoids binding another HUD over
an existing one. The PowerShell launcher was not executed during branch creation
because PowerShell was unavailable on this Mac.

For policy settings, staged Intel snapshot paths and operating limits, see
[HEURISTIC_SORTER.md](HEURISTIC_SORTER.md). For the existing camera-wall publisher,
see [CAMERA_WALL_MARKS.md](CAMERA_WALL_MARKS.md). Models, captures, virtual
environments and other ignored artifacts are not included in Git.
