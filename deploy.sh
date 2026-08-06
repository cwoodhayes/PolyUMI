#!/usr/bin/env bash
# deploy.sh - Deploy PolyUMI Pi code to the Raspberry Pi.
# Usage: ./deploy.sh <ssh_hostname>

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <ssh_hostname>"
    exit 1
fi

PI_HOST="$1"

echo "==> Generating _version.py..."
COMMIT_HASH=$(git rev-parse HEAD)
echo "COMMIT_HASH = '${COMMIT_HASH}'" > pi/polyumi_pi/_version.py

echo "==> Deploying pi/ to ${PI_HOST}..."
rsync -av --delete --mkpath \
    --exclude='.venv/' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    pi "${PI_HOST}":~/PolyUMI/

echo "==> Deploying polyumi_pi_msgs to ${PI_HOST}..."
# NB: *_pb2.py / *_pb2.pyi are deliberately NOT shipped — they are machine-specific build
# artifacts (which is also why they're gitignored), and copying this PC's copies to the Pi
# is an outright bug. Generated code carries a gencode-version stamp that must not be newer
# than the importing machine's protobuf runtime, and the two machines don't agree:
#   - this PC's ROS side runs system protobuf 4.21.12, and `colcon build` regenerates these
#     files in-place using the system's protoc 3.5.1 -> pre-3.19 gencode, which protobuf 6.x
#     REFUSES to load ("Descriptors cannot be created directly").
#   - the Pi runs protobuf 6.33.6, which needs modern gencode stamped <= its own version.
# So whichever tool ran last on the PC (colcon build vs. a uv PEP-517 build) used to decide
# what the Pi received. Instead the Pi builds its own below, against its own runtime.
rsync -av --delete --mkpath \
    --exclude='.venv/' \
    --exclude='*.egg-info/' \
    --exclude='__pycache__/' \
    --exclude='*_pb2.py' \
    --exclude='*_pb2.pyi' \
    ros2_ws/src/polyumi_pi_msgs "${PI_HOST}":~/PolyUMI/ros2_ws/src/

echo "==> Syncing Pi venv (and regenerating protobuf bindings ON the Pi)..."
# --reinstall-package polyumi-pi-msgs forces its PEP-517 build to re-run here, which is what
# regenerates the *_pb2.py excluded above (setup.py's compile_protos()). Without it, uv sees
# an already-installed editable package and skips the build, leaving the Pi with whatever
# stale bindings it had. Costs a few seconds per deploy; correctness is worth it.
ssh "${PI_HOST}" '
    set -euo pipefail
    [ -d ~/PolyUMI/pi/.venv ] || ~/.local/bin/uv venv --system-site-packages ~/PolyUMI/pi/.venv
    cd ~/PolyUMI/pi && ~/.local/bin/uv sync --no-dev --frozen --extra pi \
        --reinstall-package polyumi-pi-msgs
'

echo "==> Verifying protobuf bindings import on the Pi..."
# Fail the deploy here rather than at stream time: a gencode/runtime mismatch is an IMPORT
# error, so it takes down every polyumi-pi command, not just the one you were running.
ssh "${PI_HOST}" '
    cd ~/PolyUMI/pi && .venv/bin/python -c "
from polyumi_pi_msgs.audio_chunk_pb2 import AudioChunk
from polyumi_pi_msgs.camera_frame_pb2 import CameraFrame
import google.protobuf
print(\"    protobuf bindings OK (runtime\", google.protobuf.__version__ + \")\")
"
'

echo "==> Applying ALSA preset (UCM warnings about 'use case configuration' are harmless)..."
ssh "${PI_HOST}" "sudo alsactl restore -f ~/PolyUMI/pi/alsa_preset || true"

echo "==> Updating WM8960 ALSA state file..."
ssh "${PI_HOST}" "sudo cp ~/PolyUMI/pi/alsa_preset /etc/wm8960-soundcard/wm8960_asound.state"

echo "==> Done. Deployed commit ${COMMIT_HASH} to ${PI_HOST}."
echo "    Restart the service to pick up code changes:"
echo "      sudo systemctl restart polyumi-pi"
