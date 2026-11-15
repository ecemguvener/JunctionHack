param(
    [int]$MaxCycles = 24,
    [ValidateSet("lab-dark-mark", "patchcore")]
    [string]$InspectionMode = "lab-dark-mark",
    [string]$OpenVinoModel = "",
    [switch]$DryRun,
    [switch]$SyncOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "[1/4] Syncing updated Brickcell block_inspector + inspection/detector modules + configs to intel:/home/intel/brickcell-deploy ..." -ForegroundColor Cyan
scp `
    "$RepoRoot\src\brickcell\block_inspector\armd_bridge.py" `
    "$RepoRoot\src\brickcell\block_inspector\cogman.py" `
    "$RepoRoot\src\brickcell\block_inspector\cli.py" `
    "$RepoRoot\src\brickcell\block_inspector\hardware_cell.py" `
    intel:/home/intel/brickcell-deploy/src/brickcell/block_inspector/

scp -r `
    "$RepoRoot\src\brickcell\inspection" `
    "$RepoRoot\src\brickcell\detector" `
    intel:/home/intel/brickcell-deploy/src/brickcell/

scp `
    "$RepoRoot\configs\rgbd-viewer.json" `
    intel:/home/intel/brickcell-deploy/configs/rgbd-viewer.json

Write-Host "[2/4] Verifying remote syntax (inspection policy: $InspectionMode) ..." -ForegroundColor Cyan
ssh intel "/home/intel/brickcell-deploy/.venv/bin/python -m py_compile /home/intel/brickcell-deploy/src/brickcell/block_inspector/armd_bridge.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/cogman.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/cli.py /home/intel/brickcell-deploy/src/brickcell/block_inspector/hardware_cell.py"
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
