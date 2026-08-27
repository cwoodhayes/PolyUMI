#!/usr/bin/env bash
# serve_policy.sh - Build and run the PolyUMI diffusion-policy inference server.
#
# Host-side orchestration for the GPU workstation (rootless Docker), mirroring train_policy.sh:
# it wires the checkpoint + HF-cache mounts and the rootless-safe run flags, then runs the
# container's serve entrypoint (docker/serve.sh -> uvicorn serve_policy:app). serve_policy.py
# serves the trained policy over POST /predict_cartesian/ + /reset; the ROS-side
# policy_client_node POSTs to it. Full walkthrough in docs/training-instructions.md.
#
# NOTE: external/polyumi_diffusion_policy/docker/serve.sh is the IN-CONTAINER entrypoint and will
# NOT run on the host ("exec: uvicorn: not found") — the umi conda env only exists in the image.
# Always launch via this script (or a raw `docker run ... bash docker/serve.sh`).
#
# Usage:
#   CKPT=/abs/path/to/epoch=0070-....ckpt ./serve_policy.sh
#   CKPT=... PORT=8001 ./serve_policy.sh
#   CKPT=... CUDA_VISIBLE_DEVICES=1 ./serve_policy.sh    # pin to the second GPU
#
# serve_policy.py loads the policy on plain 'cuda', i.e. whatever CUDA_VISIBLE_DEVICES makes
# device 0. Not defaulted here: which card is quiet changes hour to hour, so check `nvidia-smi`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_DIR="${REPO_ROOT}/external/polyumi_diffusion_policy"

IMAGE="${IMAGE:-polyumi-dp}"
CKPT="${CKPT:?set CKPT=/abs/path/to/<name>.ckpt (a trained checkpoint from train_policy.sh)}"
PORT="${PORT:-8002}"

# GPU flag. --gpus all works on most setups; if it fails under rootless Docker, set
#   GPU_FLAG="--device nvidia.com/gpu=all"   (the CDI form; see training-instructions.md).
GPU_FLAG="${GPU_FLAG:---gpus all}"

# Run-as-user. Container root maps back to the host user under rootless Docker, so the bind-mounted
# HF cache is writable. On a rootful Docker host set USER_FLAG="--user $(id -u):$(id -g)" instead.
# See train_policy.sh § "output file ownership".
USER_FLAG="${USER_FLAG:---user 0:0}"

# HuggingFace cache — shared with train_policy.sh so the timm ViT encoder weights (~600 MB) aren't
# re-downloaded on every start (HOME=/tmp is ephemeral). Override with HF_CACHE_DIR.
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

# Allocate a TTY only when stdin is one, so the script also works over non-interactive SSH
# (docker run -t against a non-TTY errors with "the input device is not a TTY").
TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-t"

if [ ! -f "${CKPT}" ]; then
    echo "error: CKPT '${CKPT}' not found" >&2
    exit 1
fi

mkdir -p "${HF_CACHE_DIR}"

echo ">> building ${IMAGE} from ${FORK_DIR}"
docker build -t "${IMAGE}" "${FORK_DIR}"

echo ">> serving (checkpoint: ${CKPT}) on http://0.0.0.0:${PORT}"
# The checkpoint file is mounted directly (its filename contains '=', which is fine for a bind
# mount — only ':' would confuse -v); CKPT_PATH points at the clean in-container path. HOME and
# cache dirs point at /tmp so anything needing a writable home works regardless of the container
# uid under rootless.
# shellcheck disable=SC2086
exec docker run --rm -i ${TTY_FLAG} \
    ${GPU_FLAG} \
    ${USER_FLAG} \
    -p "${PORT}:8000" \
    -e HOME=/tmp \
    -e CUDA_VISIBLE_DEVICES \
    -e MPLCONFIGDIR=/tmp/mpl \
    -e NUMBA_CACHE_DIR=/tmp/numba \
    -e HF_HOME=/hf_cache \
    -e CKPT_PATH=/data/model.ckpt \
    -v "${CKPT}:/data/model.ckpt:ro" \
    -v "${HF_CACHE_DIR}:/hf_cache:rw" \
    "${IMAGE}" \
    bash docker/serve.sh
