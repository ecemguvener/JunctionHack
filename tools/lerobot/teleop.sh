#!/usr/bin/env bash
# One-shot leader -> follower teleoperation of the two SO-101 arms on the Intel server.
# Run on your own computer. It uses SSH to the server (see ssh.md).
#
#   tools/lerobot/teleop.sh start    Release the arms in armd and start teleop_aligned.py in tmux.
#   tools/lerobot/teleop.sh status   Show the teleop process, the arm ports and the last screen.
#   tools/lerobot/teleop.sh attach   Show the live screen (Ctrl+B then D to detach).
#   tools/lerobot/teleop.sh stop     Stop teleoperation and give the arms back to armd.
#
# Settings (environment variables):
#   TELEOP_HOST  SSH host            (default: intel)
#   TELEOP_DIR   repo on the server  (default: ~/Desktop/tarot-club-junction)
#   ALIGN_LIMIT  alignment window, degrees (default: 20)
#   MAX_STEP     follower step limit, degrees (default: 30)
set -euo pipefail

HOST="${TELEOP_HOST:-intel}"
DIR="${TELEOP_DIR:-~/Desktop/tarot-club-junction}"
ALIGN_LIMIT="${ALIGN_LIMIT:-20}"
MAX_STEP="${MAX_STEP:-30}"
ACTION="${1:-start}"

remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST" \
    "DIR=$DIR ALIGN_LIMIT=$ALIGN_LIMIT MAX_STEP=$MAX_STEP bash -s -- $1" <<'REMOTE'
set -uo pipefail
ARMD=http://127.0.0.1:8765
PORTS="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00 /dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE7044736-if00"
# A separate tmux server: an old default tmux server can lack the dialout group.
T="tmux -L teleop"

running() { pgrep -f 'teleop_aligned.py|lerobot-teleoperate' >/dev/null; }
screen() { $T capture-pane -p -t teleop 2>/dev/null | grep -v '^$' | tail -"${1:-8}"; }
armd_up() { curl -s -m3 "$ARMD/health" >/dev/null; }

case "$1" in
start)
  if running; then echo "Teleoperation runs already."; screen; exit 0; fi
  if armd_up; then
    for a in follower leader; do
      printf 'armd %s: ' "$a"; curl -s -m5 -X POST "$ARMD/arms/$a/release"; echo
    done
  else
    echo "armd does not answer. Continue without it."
  fi
  sleep 1
  for p in $PORTS; do
    [ -e "$p" ] || { echo "ERROR: port $p is missing. Examine the USB cable and the hub power."; exit 1; }
    if fuser "$p" >/dev/null 2>&1; then
      echo "ERROR: another process uses $p:"; fuser -v "$p"; exit 1
    fi
  done
  $T kill-server 2>/dev/null || true
  cd "$DIR" || { echo "ERROR: $DIR is missing on the server."; exit 1; }
  $T new-session -d -s teleop -x 120 -y 30 \
    "HF_LEROBOT_CALIBRATION=\$HOME/.cache/huggingface/lerobot/calibration ALIGN_LIMIT=$ALIGN_LIMIT MAX_STEP=$MAX_STEP \
     .venv/bin/python tools/lerobot/teleop_aligned.py 2>&1 | tee /tmp/teleop.log; exec bash"
  sleep 5
  if ! running; then echo "ERROR: teleoperation did not start. Log:"; tail -20 /tmp/teleop.log; exit 1; fi
  echo; echo "Teleoperation started. Move the LEADER until all joints show OK for 1 s."
  echo "Then the follower follows the leader (step limit $MAX_STEP deg)."; echo
  screen 6
  ;;
status)
  if running; then echo "Teleoperation: running"; else echo "Teleoperation: stopped"; fi
  for p in $PORTS; do echo "$p:"; fuser -v "$p" 2>&1 | tail -n +2 || true; done
  armd_up && { printf 'armd: '; curl -s -m3 "$ARMD/health"; echo; } || echo "armd: no answer"
  echo "--- last screen ---"; screen 6
  ;;
stop)
  if running; then
    echo "Stopping teleoperation. HOLD THE FOLLOWER: LeRobot turns off its torque."
    $T send-keys -t teleop C-c 2>/dev/null || true
    for _ in $(seq 10); do running || break; sleep 0.5; done
    running && pkill -INT -f 'teleop_aligned.py|lerobot-teleoperate'; sleep 1
  fi
  $T kill-server 2>/dev/null || true
  if armd_up; then
    for a in follower leader; do
      printf 'armd %s: ' "$a"; curl -s -m5 -X POST "$ARMD/arms/$a/acquire"; echo
    done
  else
    echo "armd does not answer. The ports are free."
  fi
  ;;
*) echo "Unknown action: $1"; exit 2 ;;
esac
REMOTE
}

case "$ACTION" in
  start|status|stop) remote "$ACTION" ;;
  attach) exec ssh -t "$HOST" tmux -L teleop attach -t teleop ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
