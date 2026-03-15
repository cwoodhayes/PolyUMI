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
rsync -av --delete \
    --exclude='.venv/' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    pi "${PI_HOST}":~

echo "==> Deploying polyumi_pi_msgs to ${PI_HOST}..."
rsync -av --delete \
    --exclude='.venv/' \
    ros2_ws/src/polyumi_pi_msgs "${PI_HOST}":~

echo "==> Done. Deployed commit ${COMMIT_HASH} to ${PI_HOST}."