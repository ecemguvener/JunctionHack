#!/usr/bin/env bash
# Install or remove the armd systemd service. Run this script on the server as intel.
#
# Usage: deploy/install.sh [install|uninstall|status] [--run]
#   install    (default) Check the setup, then show the sudo commands that install the service.
#   uninstall  Show the sudo commands that stop and remove the service.
#   status     Show the service status and the last log lines.
#
# With no --run, the script only shows the root commands.
# With --run, the script runs them with "sudo -n". The intel user has passwordless
# sudo (see ssh.md). If sudo asks for a password, "sudo -n" fails and does not wait.
set -euo pipefail

APP_DIR="/home/intel/arm-api"
UNIT_SRC="$APP_DIR/deploy/armd.service"
UNIT_DST="/etc/systemd/system/armd.service"
PY="/home/intel/miniforge3/envs/armd/bin/python"

CMD="install"
RUN=0
for arg in "$@"; do
  case "$arg" in
    install|uninstall|status) CMD="$arg" ;;
    --run) RUN=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Show the root commands, or run them if --run is given.
root_cmds() {
  if [ "$RUN" -eq 1 ]; then
    echo "Run as root:"; printf '  %s\n' "$@"
    local c
    for c in "$@"; do bash -c "$c"; done
  else
    echo "Run these commands on the server (they need sudo):"
    echo
    local joined
    joined=$(printf '%s && ' "$@"); joined=${joined% && }
    echo "  $joined"
    echo
    echo "Or run: $0 $CMD --run"
  fi
}

check_setup() {
  local ok=1
  if [ "$(id -un)" != "intel" ]; then
    echo "WARNING: run this script as the intel user (now: $(id -un))." >&2
  fi
  if [ ! -f "$UNIT_SRC" ]; then
    echo "ERROR: $UNIT_SRC is missing. Run deploy/deploy.sh on the laptop first." >&2; ok=0
  fi
  if [ ! -x "$PY" ]; then
    echo "ERROR: $PY is missing. Run deploy/deploy.sh on the laptop first." >&2; ok=0
  elif ! "$PY" -c "import armd, scservo_sdk, fastapi, uvicorn, yaml, serial" 2>/dev/null; then
    echo "ERROR: the armd env cannot import armd or its dependencies. Run deploy/deploy.sh again." >&2; ok=0
  fi
  if [ ! -f "$APP_DIR/config.yaml" ]; then
    echo "ERROR: $APP_DIR/config.yaml is missing." >&2; ok=0
  fi
  if ! id -nG intel | tr ' ' '\n' | grep -qx dialout; then
    echo "WARNING: intel is not in the dialout group. The service cannot open the serial ports." >&2
  fi
  for p in $(grep -E '^\s*port:\s*/dev/' "$APP_DIR/config.yaml" 2>/dev/null | awk '{print $2}'); do
    [ -e "$p" ] || echo "WARNING: $p is missing. That arm will be offline until you connect it." >&2
  done
  if pgrep -u intel -f 'lerobot' >/dev/null 2>&1; then
    echo "WARNING: a lerobot process runs. Only one process can open a serial port. Stop it first." >&2
  fi
  [ "$ok" -eq 1 ]
}

case "$CMD" in
  install)
    check_setup || { echo "Fix the errors, then run this script again." >&2; exit 1; }
    echo "Setup is OK."
    root_cmds \
      "sudo -n install -m 644 $UNIT_SRC $UNIT_DST" \
      "sudo -n systemctl daemon-reload" \
      "sudo -n systemctl enable --now armd"
    echo
    echo "Then check: systemctl status armd; journalctl -u armd -f; curl -s http://127.0.0.1:8765/health"
    ;;
  uninstall)
    root_cmds \
      "sudo -n systemctl disable --now armd" \
      "sudo -n rm -f $UNIT_DST" \
      "sudo -n systemctl daemon-reload"
    ;;
  status)
    systemctl status armd --no-pager || true
    echo
    journalctl -u armd -n 20 --no-pager 2>/dev/null || echo "(Cannot read the journal. Use: sudo journalctl -u armd)"
    echo
    curl -s --max-time 3 http://127.0.0.1:8765/health || echo "(No answer on http://127.0.0.1:8765/health)"
    echo
    ;;
esac
