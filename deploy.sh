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
rsync -av --delete --mkpath \
    --exclude='.venv/' \
    ros2_ws/src/polyumi_pi_msgs "${PI_HOST}":~/PolyUMI/ros2_ws/src/

echo "==> Syncing Pi venv..."
ssh "${PI_HOST}" '
    set -euo pipefail
    [ -d ~/PolyUMI/pi/.venv ] || ~/.local/bin/uv venv --system-site-packages ~/PolyUMI/pi/.venv
    cd ~/PolyUMI/pi && ~/.local/bin/uv sync --no-dev --frozen
'

echo "==> Applying ALSA preset (UCM warnings about 'use case configuration' are harmless)..."
ssh "${PI_HOST}" "sudo alsactl restore -f ~/PolyUMI/pi/alsa_preset"

echo "==> Done. Deployed commit ${COMMIT_HASH} to ${PI_HOST}."
echo "    Restart the service to pick up code changes:"
echo "      sudo systemctl restart polyumi-pi"
