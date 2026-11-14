#!/usr/bin/env bash
# Copy the station twin code to the robot laptop (Intel) and print the start command.
# It is safe to run again: rsync copies only changed files. It does not start or stop anything.
#
#   scripts/deploy_twin.sh             # copy to intel:/home/intel/brickcell-twin
#   scripts/deploy_twin.sh --dry-run   # show what rsync would copy
#   TWIN_HOST=intel TWIN_DEST=/home/intel/brickcell-twin scripts/deploy_twin.sh
set -euo pipefail

HOST="${TWIN_HOST:-intel}"
DEST="${TWIN_DEST:-/home/intel/brickcell-twin}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY=()
[[ "${1:-}" == "--dry-run" ]] && DRY=(--dry-run)

ssh -o BatchMode=yes "$HOST" "mkdir -p '$DEST'"
# src: the brickcell package; configs: twin.yaml and sim-scene.yaml; assets: brick meshes for
# perception; calibration: arm calibration and CAD alignment for the arm reader.
for dir in src configs assets calibration; do
  rsync -az --delete "${DRY[@]}" --exclude '__pycache__' --exclude '*.pyc' \
    "$ROOT/$dir/" "$HOST:$DEST/$dir/"
done

UV=/home/intel/.local/bin/uv
ENV="--offline --no-project --python 3.12 --with pyrealsense2 --with numpy"
ENV="$ENV --with opencv-python-headless --with pyyaml --with scipy"
cat <<EOF
Deployed to $HOST:$DEST

Start (on $HOST; only one process can own the D415s, stop the live-depth viewer first):
  cd $DEST && PYTHONPATH=src $UV run $ENV python -m brickcell.twin.station --arm-source none

Replay a fixture instead of the cameras:
  cd $DEST && PYTHONPATH=src $UV run $ENV python -m brickcell.twin.station \\
      --fixture /home/intel/twin-fixtures/fixture-001 --arm-source none

From this machine: ssh -N -L 8095:127.0.0.1:8095 $HOST, then open http://127.0.0.1:8095/
EOF
