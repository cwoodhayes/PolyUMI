#!/usr/bin/env bash
# deploy_gpu.sh - Deploy the training/serving half of PolyUMI to the GPU box.
# Usage: ./deploy_gpu.sh [ssh_hostname] [remote_repo_path]
#
# The GPU box runs exactly two things from this repo: train_policy.sh and serve_policy.sh, both
# of which drive the polyumi_diffusion_policy fork inside a docker image. It needs none of the
# Pi, ROS, ingest or catalog trees, so this ships the fork plus the two entrypoints rather than
# the whole 2.7 GB working copy.
#
# Companion to deploy.sh (the Pi) and fr3_session.sh's rsync of nuc/ (the NUC). Same idea: the
# remote runs THIS working copy, not whatever it last had.

set -euo pipefail

GPU_HOST="${1:-${GPU_SSH_HOST:-lamb}}"
# Left unexpanded so the REMOTE shell resolves the tilde against its own $HOME.
GPU_REPO="${2:-${GPU_REPO:-~/repos/PolyUMI}}"

echo "==> Deploying to ${GPU_HOST}:${GPU_REPO} ..."

# --delete, so a file removed here is removed there — a stale serve_obs.py on the GPU box is the
# kind of thing that produces a plausible-looking rollout against last week's frame convention.
#
# The excludes are the same classes deploy.sh documents:
#   *.pyc / __pycache__ / *.egg-info  - build artifacts, and .pyc from a different interpreter
#                                       version is worse than useless
#   .venv                             - a venv is not relocatable across machines
#   data/, dp_outputs/                - datasets and checkpoints; the GPU box GENERATES these and
#                                       they are far larger than the code. Never overwrite them
#                                       from here.
#   external/ORB_SLAM3_PolyUMI        - 105 MB of C++ for the ingest-side SLAM step; nothing on
#                                       the GPU box builds or runs it.
rsync -av --delete --mkpath \
    --exclude='.venv/' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    --exclude='*.egg-info/' \
    --exclude='.git/' \
    --exclude='data/' \
    --exclude='dp_outputs/' \
    --exclude='wandb/' \
    external/polyumi_diffusion_policy \
    "${GPU_HOST}:${GPU_REPO}/external/"

rsync -av --mkpath \
    train_policy.sh serve_policy.sh build_policy_image.sh \
    "${GPU_HOST}:${GPU_REPO}/"

# The shared protocol library and the Dockerfile that layers it in. Both ends of the inference
# protocol import polyumi_inference -- the ROS client and the policy server -- so the GPU box needs
# the same working copy the laptop has, not a copy of it.
rsync -av --delete --mkpath \
    --exclude='.venv/' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    --exclude='*.egg-info/' \
    --exclude='.pytest_cache/' \
    inference_server docker \
    "${GPU_HOST}:${GPU_REPO}/"

rsync -av --delete --mkpath \
    --exclude='__pycache__/' \
    docs "${GPU_HOST}:${GPU_REPO}/"

echo "==> Verifying the fork landed intact on ${GPU_HOST}..."
# Import-check serve_obs rather than just listing files: it is the module both entrypoints load,
# and a partial rsync shows up here as an ImportError instead of as a failed rollout later.
ssh "${GPU_HOST}" "
    set -euo pipefail
    cd ${GPU_REPO}/external/polyumi_diffusion_policy
    test -f serve_policy.py && test -f serve_obs.py
    echo '    fork present'
    test -f ${GPU_REPO}/inference_server/polyumi_inference/wire.py
    test -f ${GPU_REPO}/docker/polyumi_inference.Dockerfile
    echo '    polyumi_inference present'
"

echo "==> Done. Next:"
echo "      ssh ${GPU_HOST} 'cd ${GPU_REPO} && ./train_policy.sh'    # builds the image if absent"
echo "      ssh ${GPU_HOST} 'cd ${GPU_REPO} && CKPT=<path> ./serve_policy.sh'"
