$ErrorActionPreference = "Continue"

$RemotePy = @'
import os, signal, subprocess, time, urllib.request
res = subprocess.run(["pgrep", "-f", "deploy-cell"], capture_output=True, text=True)
pids = [int(p) for p in res.stdout.split() if p.isdigit() and int(p) != os.getpid()]
for p in pids:
    try:
        os.kill(p, signal.SIGINT)
    except ProcessLookupError:
        pass
time.sleep(0.25)
for p in pids:
    try:
        os.kill(p, signal.SIGTERM)
    except ProcessLookupError:
        pass
req = urllib.request.Request(
    "http://127.0.0.1:8765/arms/follower/stop",
    data=b'{"mode":"hold"}',
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=3) as resp:
    print("STOP_HOLD:", resp.read().decode().strip(), "stopped_pids=", pids)
'@

$RemotePy | ssh -o ConnectTimeout=4 intel "/home/intel/brickcell-deploy/.venv/bin/python -"
