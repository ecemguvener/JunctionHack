#!/usr/bin/env bash
# Deploy armd from this laptop to the Intel server.
#
# Usage: arm-api/deploy/deploy.sh [--dry-run] [--no-install]
#   --dry-run     Show what rsync would copy. Do not change the server.
#   --no-install  Copy the code only. Do not create the env or run pip.
#
# The script:
#   1. Copies arm-api/ to intel:/home/intel/arm-api/ with rsync.
#   2. Creates the conda env "armd" (Python 3.12) if it is missing.
#   3. Runs "pip install -e /home/intel/arm-api" in the env.
# You can run it again at any time. It does not start or restart the service.
# To restart the service after a deploy: ssh intel, then "sudo systemctl restart armd".
set -euo pipefail

HOST="${ARMD_HOST:-intel}"
REMOTE_DIR="/home/intel/arm-api"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15)
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=0
INSTALL=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-install) INSTALL=0 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Run a command. If it fails, wait 5 s and try again (3 tries in total).
# The network to the server is not reliable (see ssh.md).
retry() {
  local n
  for n in 1 2 3; do
    if "$@"; then return 0; fi
    if [ "$n" -lt 3 ]; then
      echo "Command failed (try $n of 3). Wait 5 s and try again." >&2
      sleep 5
    fi
  done
  echo "Command failed 3 times. Make sure that you are on the Junction2026 Wi-Fi." >&2
  return 1
}

RSYNC_ARGS=(
  -az --delete
  --exclude .venv/
  --exclude __pycache__/
  --exclude .pytest_cache/
  --exclude '*.egg-info/'
  --exclude CONTRACT.md
  -e "ssh ${SSH_OPTS[*]}"
)
[ "$DRY_RUN" -eq 1 ] && RSYNC_ARGS+=(--dry-run --itemize-changes)

echo "Copy $SRC_DIR/ to $HOST:$REMOTE_DIR/"
retry rsync "${RSYNC_ARGS[@]}" "$SRC_DIR/" "$HOST:$REMOTE_DIR/"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run: the server did not change."
  exit 0
fi
if [ "$INSTALL" -eq 0 ]; then
  echo "Code copied. Env install skipped (--no-install)."
  exit 0
fi

# The remote install runs with nohup and writes a log, so that a dropped
# connection does not stop it. This script then polls the log.
REMOTE_SCRIPT=$(cat <<'REMOTE'
set -euo pipefail
CONDA="$HOME/miniforge3/bin/conda"
PY="$HOME/miniforge3/envs/armd/bin/python"
echo "=== start $(date -Is)"
if [ ! -x "$PY" ]; then
  echo "Create the conda env armd (Python 3.12)."
  "$CONDA" create -y -n armd python=3.12
fi
"$PY" -m pip install -e /home/intel/arm-api
"$PY" -c "import armd, scservo_sdk, fastapi, uvicorn, yaml, serial; print('armd', getattr(armd, '__version__', '?'), 'import OK')"
echo "=== ARMD_DEPLOY_OK $(date -Is)"
REMOTE
)

LOG="/home/intel/armd_deploy.log"
echo "Install on $HOST (log: $LOG)."
RUN="/home/intel/armd_deploy_run.sh"
PIDFILE="/home/intel/armd_deploy.pid"
retry ssh "${SSH_OPTS[@]}" "$HOST" "cat > '$RUN'" <<<"$REMOTE_SCRIPT"
retry ssh "${SSH_OPTS[@]}" "$HOST" \
  "if [ -f '$PIDFILE' ] && kill -0 \$(cat '$PIDFILE') 2>/dev/null; then echo 'An install runs already. Wait for it.'; \
   else nohup bash '$RUN' > '$LOG' 2>&1 < /dev/null & echo \$! > '$PIDFILE'; fi"

# Poll the log until the install ends (at most 15 min).
for _ in $(seq 1 180); do
  sleep 5
  out=$(ssh "${SSH_OPTS[@]}" "$HOST" "tail -n 3 '$LOG'; kill -0 \$(cat '$PIDFILE') 2>/dev/null && echo __RUNNING__ || echo __STOPPED__" 2>/dev/null) || continue
  if grep -q ARMD_DEPLOY_OK <<<"$out"; then
    echo "$out" | grep -v __STOPPED__
    echo "Deploy done. Restart the service if it runs: sudo systemctl restart armd"
    exit 0
  fi
  if grep -q __STOPPED__ <<<"$out"; then
    echo "The install stopped with an error. Last lines of $LOG:" >&2
    ssh "${SSH_OPTS[@]}" "$HOST" "tail -n 30 '$LOG'" >&2 || true
    exit 1
  fi
done
echo "The install did not end in 15 min. Read $LOG on the server." >&2
exit 1
