param(
    [int]$MaxCycles = 24,
    [ValidateSet("lab-dark-mark", "patchcore", "patchcore-plate")]
    [string]$InspectionMode = "patchcore-plate",
    [string]$OpenVinoModel = "",
    [switch]$DryRun,
    [switch]$SyncOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[1/4] Syncing updated Brickcell block_inspector + inspection/detector modules + configs to intel:/home/intel/brickcell-deploy ..." -ForegroundColor Cyan
scp `
    "$RepoRoot\src\brickcell\block_inspector\armd_bridge.py" `
    "$RepoRoot\src\brickcell\block_inspector\camera_adapters.py" `
    "$RepoRoot\src\brickcell\block_inspector\cogman.py" `
    "$RepoRoot\src\brickcell\block_inspector\cli.py" `
    "$RepoRoot\src\brickcell\block_inspector\frax_ik.py" `
    "$RepoRoot\src\brickcell\block_inspector\hardware_cell.py" `
    "$RepoRoot\src\brickcell\block_inspector\runtime_bus.py" `
    "$RepoRoot\src\brickcell\block_inspector\runtime_nodes.py" `
    intel:/home/intel/brickcell-deploy/src/brickcell/block_inspector/

scp `
    "$RepoRoot\scripts\publish_mark_inspection.py" `
    "$RepoRoot\scripts\publish_camui_detection.py" `
    intel:/home/intel/brickcell-deploy/scripts/

scp -r `
    "$RepoRoot\services\bowl_circle_tracker" `
    "$RepoRoot\services\wrist_anomaly_detector" `
    intel:/home/intel/brickcell-deploy/services/

scp -r `
    "$RepoRoot\src\brickcell\inspection" `
    "$RepoRoot\src\brickcell\detector" `
    intel:/home/intel/brickcell-deploy/src/brickcell/

scp `
    "$RepoRoot\configs\rgbd-viewer.json" `
    intel:/home/intel/brickcell-deploy/configs/rgbd-viewer.json

scp `
    "$RepoRoot\camera-ui\camui.py" `
    "$RepoRoot\camera-ui\index.html" `
    intel:/home/intel/brickcell-deploy/camera-ui/

scp `
    "$RepoRoot\camera-ui\camui.py" `
    "$RepoRoot\camera-ui\index.html" `
    intel:/home/intel/camera-ui/

Write-Host "[2/4] Verifying remote syntax (inspection policy: $InspectionMode) ..." -ForegroundColor Cyan
ssh intel "/home/intel/brickcell-deploy/.venv/bin/python -m py_compile /home/intel/brickcell-deploy/src/brickcell/block_inspector/armd_bridge.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/camera_adapters.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/cogman.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/cli.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/frax_ik.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/hardware_cell.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/runtime_bus.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/runtime_nodes.py /home/intel/brickcell-deploy/services/bowl_circle_tracker/bowl_circle_node.py /home/intel/brickcell-deploy/services/wrist_anomaly_detector/wrist_anomaly_node.py /home/intel/brickcell-deploy/scripts/publish_mark_inspection.py /home/intel/brickcell-deploy/scripts/publish_camui_detection.py"
if ($LASTEXITCODE -ne 0) {
    throw "Remote syntax verification failed; deployment was not launched."
}

Write-Host "[3/4] Ensuring Station Twin Dashboard (:8096) is focused on the Intel bench screen ..." -ForegroundColor Cyan
@'
import os, subprocess
env = dict(os.environ, DISPLAY=":0")
subprocess.Popen(
    ["firefox", "http://127.0.0.1:8096/#src=real&defects=1&focus=1&scope=all"],
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
'@ | ssh intel "/home/intel/brickcell-deploy/.venv/bin/python -"

if ($SyncOnly) {
    Write-Host "[4/4] SyncOnly complete. Ready to launch deploy-cell on intel!" -ForegroundColor Green
    exit 0
}

$ExtraArgs = "--max-cycles $MaxCycles --inspection-mode $InspectionMode --rgbd-viewer-config /home/intel/brickcell-deploy/configs/rgbd-viewer.json"
if ($DryRun) {
    $ExtraArgs += " --dry-run"
}
if ($OpenVinoModel -ne "") {
    $ExtraArgs += " --openvino-model `"$OpenVinoModel`""
}

Write-Host "[4/4] Launching live physical cell deployment on intel ($ExtraArgs) ..." -ForegroundColor Green
ssh intel "cd /home/intel/brickcell-deploy && PYTHONPATH=src /home/intel/brickcell-deploy/.venv/bin/python -m brickcell.block_inspector.cli deploy-cell $ExtraArgs"
