#!/usr/bin/env bash
# deploy_lamb.sh - Push this working copy to the GPU box, and build what runs there.
# Usage: ./deploy_lamb.sh [ssh_hostname] [remote_repo_path]
#
# lamb runs BOTH halves of inference — the ROS client (policy_client_node, the camera, Foxglove)
# and the policy server (serve_policy.sh, the diffusion-policy fork in Docker) — plus training.
# So it gets the whole tree rather than a curated subset, and the three build steps that a plain
# rsync leaves stale.
#
# Companion to deploy.sh (the Pi) and fr3_session.sh's rsync of nuc/ (the NUC). Same idea: the
# remote runs THIS working copy, not whatever it last had. fr3_session.sh calls this; run it by
# hand when you only want to push code for a training run.

set -euo pipefail

HOST="${1:-${ROS_SSH_HOST:-lamb}}"
# Left unexpanded so the REMOTE shell resolves the tilde against its own $HOME.
REPO="${2:-${ROS_REPO:-~/repos/PolyUMI}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Syncing repo to ${HOST}:${REPO} ..."
# --delete, so a file removed here is removed there — a stale serve_obs.py on lamb is the kind of
# thing that produces a plausible-looking rollout against last week's frame convention.
#
# data/, recordings/ and wandb/ are lamb's OUTPUT — data/ holds dp_outputs, i.e. every checkpoint.
# rsync protects excluded paths from --delete, which is the only reason they survive this.
# They are anchored with a leading slash because an unanchored 'data/' also matches at depth, e.g.
# a fork's own package directory (external/polyumi_vista_policy/vista/data/).
# external/ORB_SLAM3_PolyUMI is 2 GB of ingest-side C++ that nothing on lamb runs; the
# policy forks under external/ DO ship.
rsync -a --delete --mkpath \
    --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
    --exclude='.venv/' --exclude='/recordings/' --exclude='/data/' --exclude='/wandb/' \
    --exclude='external/ORB_SLAM3_PolyUMI/' \
    --exclude='ros2_ws/build/' --exclude='ros2_ws/install/' --exclude='ros2_ws/log/' \
    "${HERE}/" "${HOST}:${REPO}/"

# The Vista fork is optional (private repo, may be uninitialised). An uninitialised submodule is
# an EMPTY directory that rsync faithfully recreates on the remote, so the "do we have it at all"
# question has to be answered HERE -- remotely, a missing fork and a half-synced one look alike.
VISTA_CHECK=""
if [ -f "${HERE}/external/polyumi_vista_policy/Dockerfile" ]; then
    VISTA_CHECK="
    test -f ${REPO}/external/polyumi_vista_policy/Dockerfile
    test -f ${REPO}/external/polyumi_vista_policy/scripts/train_day0suite.sh
    echo '    vista fork present'"
fi

# Import-check the fork rather than just listing files: serve_obs is the module both entrypoints
# load, so a partial rsync shows up here instead of as a failed rollout later.
ssh "${HOST}" "
    set -euo pipefail
    test -f ${REPO}/external/polyumi_diffusion_policy/serve_policy.py
    test -f ${REPO}/external/polyumi_diffusion_policy/serve_obs.py
    test -f ${REPO}/inference_server/polyumi_inference/wire.py
    test -f ${REPO}/docker/polyumi_inference.Dockerfile
    echo '    fork + polyumi_inference present'${VISTA_CHECK}
"

# colcon COPIES sources into install/, so an edited node keeps running the old code until you
# rebuild — no error, no clue. VIRTUAL_ENV unset so the build uses the system python (see CLAUDE.md).
echo "==> Building polyumi_ros2 on ${HOST} ..."
ssh -o ConnectTimeout=10 "${HOST}" \
    "unset VIRTUAL_ENV; cd ${REPO}/ros2_ws && source /opt/ros/kilted/setup.bash \
     && colcon build --packages-select polyumi_ros2"

# policy_client_node imports polyumi_inference directly (CLAUDE.md, "The Inference Protocol Lives
# in One Library"). --no-deps so numpy/requests keep coming from apt via rosdep rather than pip
# shadowing the system numpy the rest of the ROS stack links against.
echo "==> Installing polyumi_inference for the ROS node on ${HOST} ..."
ssh -o ConnectTimeout=10 "${HOST}" \
    "cd ${REPO} && pip install --user --break-system-packages --no-deps -e inference_server/"

echo "==> Done. Next:"
echo "      ./fr3_session.sh                                          # inference"
echo "      ssh ${HOST} 'cd ${REPO} && ./train_policy.sh'             # training"
