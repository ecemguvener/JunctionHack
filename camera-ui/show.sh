#!/usr/bin/env bash
# Runs on the Intel laptop. Opens the camera wall full screen (Firefox kiosk) on the
# laptop display. A separate Firefox profile is used, so other Firefox windows do not change.
# Usage: ~/camera-ui/show.sh [stop]
set -euo pipefail

URL="http://127.0.0.1:8093/"
PROFILE="$HOME/snap/firefox/common/camui-profile"
PIDFILE="$HOME/.camui-kiosk.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")"
  sleep 1
fi
rm -f "$PIDFILE"
[ "${1:-}" = "stop" ] && { echo "Kiosk stopped."; exit 0; }

uid="$(id -u)"
export XDG_RUNTIME_DIR="/run/user/$uid"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
if [ -S "$XDG_RUNTIME_DIR/wayland-0" ]; then
  export WAYLAND_DISPLAY=wayland-0 MOZ_ENABLE_WAYLAND=1
else
  export DISPLAY=:0
fi

mkdir -p "$PROFILE"
setsid nohup firefox --new-instance --profile "$PROFILE" --kiosk "$URL" \
  > "$HOME/camui-kiosk.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "Kiosk started (pid $(cat "$PIDFILE")). Push Alt+F4 on the laptop to close it."
