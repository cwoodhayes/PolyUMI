# Training the visuomotor diffusion policy

This is the step **after** ingest and preprocessing. The full pipeline is:

```
pingest fetch → process-all → pp (preprocessing steps 1–5) → export-dp → TRAIN (this doc)
```

Training runs a fork of [UMI](https://github.com/real-stanford/universal_manipulation_interface)
(`external/polyumi_diffusion_policy`) inside a **Docker** image, so it needs no host conda
(which fights the ROS install) and is portable to the GPU workstation. The same image serves
both training and inference — the serving environment must match the training one because
checkpoints are dill-pickled and unpickle against the exact dependency tree.

## Prerequisites

- A **GPU workstation** with an NVIDIA GPU. (Training a diffusion policy with a timm vision
  encoder wants meaningfully more than the laptop's ~4 GB.)
- **Docker** with the NVIDIA container toolkit. Ours runs **rootless** (no sudo) — see the
  gotchas below.
- An **exported dataset**: `pingest export-dp <scene> --output <name>.zarr.zip`. Copy the
  `.zarr.zip` to the workstation.
- A **Weights & Biases** API key (`WANDB_API_KEY`) if you want online logging. `WANDB_ENTITY`
  and `WANDB_PROJECT` are optional — unset, they default to your wandb login's default entity
  and the `polyumi` project.

## Quick start

From the PolyUMI repo root on the workstation:

```bash
# Build the image and run a short overfit smoke test (loss should drop):
DATASET=/abs/path/to/your.zarr.zip \
WANDB_API_KEY=... \
./train_policy.sh training.num_epochs=3 task.dataset.val_ratio=0 logging.mode=offline

# Full run, logging to a specific entity/project instead of the defaults:
DATASET=/abs/path/to/your.zarr.zip \
WANDB_API_KEY=... \
WANDB_ENTITY=my-team \
WANDB_PROJECT=my-project \
./train_policy.sh
```

`train_policy.sh` builds `external/polyumi_diffusion_policy` and runs it with the rootless-safe
flags and the dataset/output mounts. Any extra arguments pass straight through to `train.py` as
**Hydra overrides** (`training.num_epochs=...`, `logging.mode=offline`, `task.dataset_path=...`,
etc). Outputs (checkpoints, hydra logs) land in `data/dp_outputs/` by default (`OUTPUT_DIR`).

`WANDB_ENTITY`/`WANDB_PROJECT` are forwarded to the container only if set in the calling shell
(the config resolves them via `${oc.env:WANDB_ENTITY,null}` / `${oc.env:WANDB_PROJECT,polyumi}`),
so leaving them unset falls back to your wandb login's default entity and the `polyumi` project
— it does not fall back to a hard-coded team.

## The two entrypoints

The image (`polyumi-dp`) has one env and two thin wrappers, both in the fork's `docker/`:

| Command | Wrapper | Purpose |
|---|---|---|
| `bash docker/train.sh` *(default)* | runs `python train.py --config-name=train_diffusion_unet_timm_polyumi_workspace` | training |
| `bash docker/serve.sh` | runs `uvicorn serve_policy:app` on `:8000` | inference server (see below) |

## Rootless Docker gotchas

Rootless is fine for this workload, but five things differ from rootful Docker. `train_policy.sh`
handles the ones it can and exposes the rest as env vars.

1. **GPU flag.** The script defaults to `--gpus all`. If that fails under rootless, use the CDI
   form: `GPU_FLAG="--device nvidia.com/gpu=all" ./train_policy.sh ...`. **Verify GPU visibility
   first** (below) — this is the most common snag.
2. **Output file ownership.** Rootless remaps container UIDs. The script defaults to
   `USER_FLAG="--user 0:0"` (container root), which under rootless Docker maps back to *your*
   host user — so the bind-mounted output dir is writable and checkpoints come back owned by
   you. (Running as the image's default `mambauser` instead maps to a subuid that can't write
   your dir — the failure you'd otherwise hit.) On a **rootful** Docker host, override with
   `USER_FLAG="--user $(id -u):$(id -g)"`. The script also sets `HOME=/tmp` and cache dirs so
   nothing needs a writable home regardless.
3. **Shared memory.** The PyTorch DataLoader needs more than Docker's 64 MB default `/dev/shm`
   or workers crash with a bus error. The script sets `--shm-size=8g` (override with `SHM_SIZE`).
   Do **not** use `--ipc=host` under rootless.
4. **Networking (inference).** Publish with `-p 8000:8000`. Do **not** use `--network host` —
   under rootless it shares the rootless net namespace, not the real host.
5. **Storage / no registry.** Build on the workstation from the fork checkout; rootless image
   storage lives under your home. Pruning the fork (a later task) shrinks the image.

## Verification

Run these in order; each isolates the next failure.

```bash
# 1. GPU visible (do this first — the likeliest rootless snag):
docker build -t polyumi-dp external/polyumi_diffusion_policy
docker run --rm --gpus all polyumi-dp nvidia-smi          # or --device nvidia.com/gpu=all
docker run --rm --gpus all polyumi-dp \
    micromamba run -n umi python -c "import torch; print(torch.cuda.is_available())"   # -> True

# 2. Dataset loads under UmiDataset (validates the exporter's schema against the real reader):
docker run --rm -v /abs/path/to/your.zarr.zip:/data/dataset.zarr.zip:ro polyumi-dp \
    micromamba run -n umi python -c "
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('eval', eval, replace=True)  # as train.py does, for latency_steps
from diffusion_policy.dataset.umi_dataset import UmiDataset
# Nest under 'task' so the \${task.*} interpolations in the config resolve.
cfg = OmegaConf.create({'task': OmegaConf.load('diffusion_policy/config/task/polyumi.yaml')})
ds = UmiDataset(shape_meta=cfg.task.shape_meta, dataset_path='/data/dataset.zarr.zip',
                pose_repr=cfg.task.pose_repr, cache_dir=None)
print('episodes:', ds.replay_buffer.n_episodes, 'obs keys:', sorted(ds[0]['obs'].keys()))
"

# 3. Overfit smoke — loss must descend:
DATASET=/abs/path/to/your.zarr.zip ./train_policy.sh \
    training.num_epochs=5 task.dataset.val_ratio=0 logging.mode=offline
```

Watch `train_action_mse_error` (and `train_loss`) descend in the console or in W&B (project
`polyumi`). The checkpointed weights are the **EMA** model (`workspace.ema_model`), not the raw
`model` — evaluation loads the EMA weights.

## Inference (after you have a checkpoint)

The same image serves the policy over the exact HTTP contract the ROS-side
`policy_client_node` already speaks to `inference_server/dummy_server.py`:

```bash
docker run --rm -it --gpus all --user 0:0 -p 8000:8000 \
    -e HOME=/tmp -e HF_HOME=/hf_cache \
    -e CKPT_PATH='/data/checkpoints/epoch=0070-train_loss=0.021.ckpt' \
    -v /abs/path/to/checkpoints:/data/checkpoints:ro \
    -v "${HOME}/.cache/huggingface:/hf_cache:rw" \
    polyumi-dp bash docker/serve.sh
```

The `-v ...huggingface:/hf_cache` mount + `HF_HOME=/hf_cache` persist the timm ViT encoder
weights (~600 MB) so the server doesn't re-download them on every start — same cache
`train_policy.sh` uses. `--gpus all` and `--user 0:0` mirror the training flags (GPU access;
rootless-safe uid). Single-quote the `CKPT_PATH` value — checkpoint filenames contain `=`.

`policy_client_node` then POSTs to `http://<workstation>:8000/predict_cartesian/` — the same
call it makes to the dummy server today, so nothing changes on the ROS side but the URL.

> **`serve_policy.py` is implemented** (loads `ema_model`, translates the wire obs to UMI's keys,
> converts the relative action chunk back to absolute EEF poses). One extra endpoint: the client
> must `POST /reset {agent_pos: [8]}` at the start of each rollout to set the episode-start pose
> (used by `robot0_eef_rot_axis_angle_wrt_start`); absent a reset it falls back to the current
> pose with a warning. Wiring `policy_client_node` to it (image→224, URL, calling `/reset`) is the
> remaining step — see [franka-inference-bringup.md](franka-inference-bringup.md) Phase 3.

## What is out of scope here

Calibration (`T_gopro_to_hand`, `latency.gopro`, the `eef_frame` static TF) is a separate ticket
and does **not** gate training: the task config sets `dataset_frequeny: 0`, which zeroes every
`latency_steps`, so the latency numbers are inert during training. A policy trained now will not
*deploy* correctly until calibration lands, but its loss curve is meaningful.
