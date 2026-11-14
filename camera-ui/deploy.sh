#!/usr/bin/env bash
# Deploy the camera wall from this computer to the Intel laptop.
#
# Usage: camera-ui/deploy.sh [--dry-run] [--no-show]
#   --dry-run  Show what rsync would copy. Do not change the server.
#   --no-show  Do not open the kiosk window on the laptop display.
#
# The script:
#   1. Copies camera-ui/ to intel:/home/intel/camera-ui/ with rsync.
#   2. Installs and restarts the systemd user service "camui" (127.0.0.1:8093).
#   3. Opens the page full screen on the laptop display (show.sh).
# The program uses the Python standard library only. It installs no packages.
set -euo pipefail

HOST="${CAMUI_HOST:-intel}"
REMOTE_DIR="/home/intel/camera-ui"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15)
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
SHOW=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-show) SHOW=0 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
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

RSYNC_ARGS=(-az --delete --exclude __pycache__/ -e "ssh ${SSH_OPTS[*]}")
[ "$DRY_RUN" -eq 1 ] && RSYNC_ARGS+=(--dry-run --itemize-changes)

echo "Copy $SRC_DIR/ to $HOST:$REMOTE_DIR/"
retry rsync "${RSYNC_ARGS[@]}" "$SRC_DIR/" "$HOST:$REMOTE_DIR/"
if [ "$DRY_RUN" -eq 1 ]; then echo "Dry run: the server did not change."; exit 0; fi

echo "Install and restart the user service camui."
retry ssh "${SSH_OPTS[@]}" "$HOST" "set -e
  mkdir -p ~/.config/systemd/user
  install -m 644 $REMOTE_DIR/camui.service ~/.config/systemd/user/camui.service
  systemctl --user daemon-reload
  systemctl --user enable camui >/dev/null 2>&1
  systemctl --user restart camui
  for i in \$(seq 1 20); do
    curl -fsS http://127.0.0.1:8093/api/status >/dev/null 2>&1 && break
    sleep 0.5
  done
  curl -fsS http://127.0.0.1:8093/api/status"
echo

if [ "$SHOW" -eq 1 ]; then
  retry ssh "${SSH_OPTS[@]}" "$HOST" "$REMOTE_DIR/show.sh"
fi
echo "Done. On the laptop: http://127.0.0.1:8093/  From here: ssh -N -L 8093:localhost:8093 $HOST"
