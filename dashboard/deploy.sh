#!/usr/bin/env bash
# Deploy the station twin dashboard from this computer to the Intel laptop.
#
# Usage: dashboard/deploy.sh [--dry-run]
#   --dry-run  Show what rsync would copy. Do not change the server.
#
# The script:
#   1. Copies dashboard/ (without tests) to intel:/home/intel/brickcell-dashboard/ with rsync.
#   2. Installs and restarts the systemd user service "twin-dashboard" (127.0.0.1:8096).
# It does not stop or change any other process. It reads from the live depth service (:8094)
# and the MuJoCo sim server (:18081); start those separately.
set -euo pipefail

HOST="${DASHBOARD_HOST:-intel}"
REMOTE_DIR="/home/intel/brickcell-dashboard"
PORT=8096
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15)
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# The network to the server is not reliable (see ssh.md). Try 3 times.
retry() {
  local n
  for n in 1 2 3; do
    if "$@"; then return 0; fi
    if [ "$n" -lt 3 ]; then echo "Command failed (try $n of 3). Wait 5 s." >&2; sleep 5; fi
  done
  echo "Command failed 3 times. Make sure that you are on the Junction2026 Wi-Fi." >&2
  return 1
}

RSYNC_ARGS=(-az --delete --exclude __pycache__/ --exclude tests/ -e "ssh ${SSH_OPTS[*]}")
[ "$DRY_RUN" -eq 1 ] && RSYNC_ARGS+=(--dry-run --itemize-changes)

echo "Copy $SRC_DIR/ to $HOST:$REMOTE_DIR/"
retry rsync "${RSYNC_ARGS[@]}" "$SRC_DIR/" "$HOST:$REMOTE_DIR/"
if [ "$DRY_RUN" -eq 1 ]; then echo "Dry run: the server did not change."; exit 0; fi

echo "Install and restart the user service twin-dashboard."
retry ssh "${SSH_OPTS[@]}" "$HOST" "set -e
  mkdir -p ~/.config/systemd/user
  install -m 644 $REMOTE_DIR/dashboard.service ~/.config/systemd/user/twin-dashboard.service
  systemctl --user daemon-reload
  systemctl --user enable twin-dashboard >/dev/null 2>&1
  systemctl --user restart twin-dashboard
  for i in \$(seq 1 40); do
    curl -fsS http://127.0.0.1:$PORT/api/status >/dev/null 2>&1 && break
    sleep 0.5
  done
  curl -fsS http://127.0.0.1:$PORT/api/status | head -c 600"
echo
echo "Done. On the laptop: http://127.0.0.1:$PORT/  From here: ssh -N -L $PORT:127.0.0.1:$PORT $HOST"
