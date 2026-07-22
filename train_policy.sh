#!/usr/bin/env bash
# train_policy.sh - Build and run the PolyUMI diffusion-policy training container.
#
# Runs on the GPU workstation (rootless Docker). The image is defined in the fork submodule
# (external/polyumi_diffusion_policy); this script is the PolyUMI-side orchestration: it wires
# up the dataset/output mounts and the rootless-safe run flags. Full walkthrough, including the
# rootless gotchas, is in docs/training-instructions.md.
#
# Usage:
#   DATASET=/abs/path/to/exported.zarr.zip WANDB_API_KEY=... ./train_policy.sh
#   # extra args pass through to train.sh as Hydra overrides, e.g. a quick overfit smoke:
#   DATASET=... ./train_policy.sh training.num_epochs=3 task.dataset.val_ratio=0 logging.mode=offline
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_DIR="${REPO_ROOT}/external/polyumi_diffusion_policy"

IMAGE="${IMAGE:-polyumi-dp}"
DATASET="${DATASET:?set DATASET=/abs/path/to/exported.zarr.zip (produced by 'pingest export-dp')}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/dp_outputs}"

# GPU flag. --gpus all works on most setups; if it fails under rootless Docker, set
#   GPU_FLAG="--device nvidia.com/gpu=all"   (the CDI form; see training-instructions.md).
GPU_FLAG="${GPU_FLAG:---gpus all}"

# Shared memory for the PyTorch DataLoader. Docker's 64 MB default crashes workers.
SHM_SIZE="${SHM_SIZE:-8g}"

# Run-as-user. Defaults to container root, which under rootless Docker maps back to the host
# user — so the bind-mounted output dir is writable and the checkpoints it writes come back
# owned by you (running as the image's default mambauser instead maps to a subuid and can't
# write your dir). On a rootful Docker host set USER_FLAG="--user $(id -u):$(id -g)" instead.
# See training-instructions.md § "output file ownership".
USER_FLAG="${USER_FLAG:---user 0:0}"

# Allocate a TTY only when stdin is one, so the script also works over non-interactive SSH
# (docker run -t against a non-TTY errors with "the input device is not a TTY").
TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-t"

if [ ! -f "${DATASET}" ]; then
    echo "error: DATASET '${DATASET}' not found" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo ">> building ${IMAGE} from ${FORK_DIR}"
docker build -t "${IMAGE}" "${FORK_DIR}"

echo ">> training (dataset: ${DATASET}, output: ${OUTPUT_DIR})"
# HOME and cache dirs point at /tmp so wandb/matplotlib/numba can write regardless of which uid
# the container ends up running as under rootless. The dataset mounts to the path the polyumi
# task config defaults to (/data/dataset.zarr.zip); outputs land in the bind-mounted dir.
# shellcheck disable=SC2086
exec docker run --rm -i ${TTY_FLAG} \
    ${GPU_FLAG} \
    ${USER_FLAG} \
    --shm-size="${SHM_SIZE}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_ENTITY \
    -e WANDB_PROJECT \
    -e HOME=/tmp \
    -e MPLCONFIGDIR=/tmp/mpl \
    -e NUMBA_CACHE_DIR=/tmp/numba \
    -v "${DATASET}:/data/dataset.zarr.zip:ro" \
    -v "${OUTPUT_DIR}:/app/data/outputs:rw" \
    "${IMAGE}" \
    bash docker/train.sh "$@"
