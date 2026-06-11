# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolyUMI is a multimodal data collection system for robot imitation learning. See [README.md](README.md) for a full description, architecture diagrams, and usage instructions.

## Common Commands

### Linting
```bash
ruff check .
ruff format .
```

### Tests
```bash
cd pi
pytest test/files/
# Single test file:
pytest test/files/test_session.py
```

When running ingest-side pytest commands in this workspace, disable pytest plugin autoload to avoid ROS-side import side effects from system site packages:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest ingest/test/test_preproc.py
```

### Deploy to Pi
```bash
./deploy.sh <ssh_hostname>   # rsync pi/ + polyumi_pi_msgs to Pi, embeds git hash in _version.py
```

### Pi (run on device)
```bash
polyumi-pi stream
polyumi-pi record-episode --fps 10 --robot polyumi_gripper --task <task_name>
polyumi-pi start-scene --robot polyumi_gripper --task <task_name>
polyumi-pi --help   # full command list
```

### ROS2 (host PC)
```bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r --rosdistro kilted
colcon build && source install/setup.bash
ros2 launch polyumi_ros2 stream_demo.launch.xml
```

#### FR3 arm (split topology)
The Franka **FR3** is driven from the **NUC** (Ubuntu 22.04, ROS2 Humble, the
Franka stack); the laptop (Ubuntu 24.04, ROS2 Kilted) runs PolyUMI's nodes,
camera, Foxglove, and `policy_client_node`. They interoperate over **CycloneDDS**
(domain 0, `10.0.0.x` link, unicast peers). On the laptop, before launching:
```bash
source setup_franka_env.sh   # RMW=cyclonedds, domain 0, CYCLONEDDS_URI, 10.0.0.1 on enp0s31f6
```
On the NUC: `fr3-bringup` + `fr3-arm-controller`. Full reference and the exact
environment assumptions live in [docs/crb-fr3-inference.md](docs/crb-fr3-inference.md).

### Ingest (host PC)
```bash
pingest --help
pingest fetch --host <hostname> --latest
pingest process-all --force
```

## Key Modules

- **`pi/polyumi_pi/main.py`** — Typer CLI; entry point for all Pi operations (`polyumi-pi`)
- **`pi/polyumi_pi/cam_streamer.py`** / **`audio_streamer.py`** — run in separate processes; communicate stats back via `multiprocessing.Pipe`
- **`pi/polyumi_pi/files/session.py`** — `SessionFiles` manages `metadata.json`, JPEG frame storage, and WAV audio
- **`pi/polyumi_pi/files/scene.py`** — `SceneFiles` groups one or more sessions under a shared scene directory
- **`pi/polyumi_pi/gopro/`** — GoPro integration via open-gopro SDK
- **`ros2_ws/src/polyumi_pi_msgs/`** — Protobuf definitions (`CameraFrame`, `AudioChunk`) with nanosecond timestamps; generated `*_pb2.py` files live alongside `.proto` sources
- **`ros2_ws/src/polyumi_ros2/`** — ROS2 package; `pi_receiver_node.py` bridges ZMQ → ROS2 topics
- **`ingest/polyumi_ingest/main.py`** — `pingest` CLI; fetches sessions from Pi via tar-over-SSH, builds pzarr working-format stores, and archives scenes to zip

## Session Data Layout
```
~/recordings/
└── scene_YYYY-MM-DD_hh-mm-ss_XXXX/
    └── session_YYYY-MM-DD_hh-mm-ss/
        ├── metadata.json
        ├── video/frame_000001.jpg ...
        └── audio.wav
```

## Package Management
This is a `uv` workspace. `ingest/` is the only workspace member. `pi/` is referenced as an editable path source (`tool.uv.sources`) so `polyumi_pi` is importable in the workspace venv, but it is not a member — it has its own `pi/.venv` managed separately for the Pi. `inference_server/` is also deliberately **not** a member: it is meant to run isolated on a standalone GPU/inference machine, so it keeps its own minimal `inference_server/.venv` (just fastapi/uvicorn/numpy) — run it with `cd inference_server && uv run dummy-server`. Run `uv sync` at the root for PC-side dev dependencies. The `pi/` package requires `--system-site-packages` on the Pi for `picamera2`/`sounddevice`.

## Running Commands in the Right Environment

Always prefix Python and tool invocations with `uv run` from the repo root — never use bare `python`, `pip`, or `ruff`:

```bash
uv run ruff check .
uv run ruff format .
uv run python -c "import polyumi_ingest"   # ingest package
uv run pytest ...
```

`uv` selects the correct workspace venv automatically. Bare `python` / `pip` will pick up the wrong venv (e.g. `pi/.venv`) and produce "module not found" errors or install into the wrong place.

**If `uv run` fails by trying to rebuild `lgpio` (Pi-only, needs `swig`):** this happens when `VIRTUAL_ENV` points at `pi/.venv` (e.g. set by a parent shell). Don't try to install swig — instead run with the already-built root venv:

```bash
unset VIRTUAL_ENV && .venv/bin/python -c "..."
unset VIRTUAL_ENV && .venv/bin/ruff check ...
# or for ruff-only:
unset VIRTUAL_ENV && uvx ruff check ...
```

The root `.venv` already has `polyumi_ingest` and its deps installed; bypassing `uv run` skips dependency resolution (which is what pulls in the unbuildable `lgpio` transitive).

**Running `colcon build` / `ros2` from a non-interactive (or zsh) shell:** sourcing
`/opt/ros/kilted/setup.bash` directly under zsh can fail with
`no such file or directory: .../ros2_ws/setup.sh` and exit 127 — the ROS setup
chain mis-resolves relative paths there. Also `VIRTUAL_ENV` pointing at `pi/.venv`
interferes. Run the build inside an explicit `bash -c`, with `VIRTUAL_ENV` unset:

```bash
unset VIRTUAL_ENV; bash -c 'cd ros2_ws && source /opt/ros/kilted/setup.bash && colcon build --packages-select polyumi_ros2'
# ros2 commands likewise: also source install/setup.bash inside the same bash -c
```

## Testing SLAM

The ORB-SLAM3 step (`OrbSlam3Step`, preprocessing step 2) uses the
`external/ORB_SLAM3_PolyUMI` git submodule by default — a PolyUMI fork of
Chi-Cheng Chang's ORB-SLAM3 fork, with additional patches (atlas-load activates
the loaded map, null guards in LocalMapping, shutdown wait-for-threads,
`ReconstructH` `vP3D` assignment, etc.) and our two custom binaries
(`mono_inertial_gopro_vi_polyumi` for mapping, `mono_inertial_gopro_vi_localize`
for localization) living in `Examples/Monocular-Inertial/`.

After a fresh clone, init the submodule and build it:

```bash
git submodule update --init --recursive
cd external/ORB_SLAM3_PolyUMI && bash build.sh
```

The build script builds Pangolin in-tree (`Thirdparty/Pangolin`) and passes
`CMAKE_PREFIX_PATH` so ORB-SLAM3's `find_package(Pangolin)` finds it; nothing extra to set up.

No env vars are required for the in-repo install — `OrbSlam3Step` resolves
`external/ORB_SLAM3_PolyUMI` from the slam_step.py source location.
Override `ORB_SLAM3_DIR` / `ORB_SLAM3_BIN_SUBDIR` only if you want to point at
an out-of-tree build.

Run the SLAM step on a single scene:
```bash
pingest pp 2 --scene recordings/scene_YYYY-MM-DD_hh-mm-ss_XXXX
# --force to re-run if already marked complete
pingest pp 2 --scene recordings/scene_YYYY-MM-DD_hh-mm-ss_XXXX --force
```

Test scene: `recordings/scene_2026-05-12_21-36-44_7985` — has one MAPPING episode,
no EPISODE sessions. Step will build the map and warn about missing episodes; that's expected.

**Camera model:** The YAML at `ingest/config/gopro_hero12_slam.yaml` currently uses
`DoubleSphere` (from the first calibration run), but this ORB-SLAM3 build only supports
`Pinhole` and `KannalaBrandt8`. A recalibration with `--camera_model=FISHEYE` in the
OpenImuCameraCalibrator Docker container is needed before map building will succeed.
`FISHEYE` in OpenImuCameraCalibrator = `KannalaBrandt8` in ORB-SLAM3 (same Kannala-Brandt
4-parameter model; output fields `radial_distortion_1..4` → `Camera.k1..k4`).

Recalibration command (corners already extracted, so this is fast):
```bash
# inside the OpenImuCameraCalibrator Docker container
python python/run_gopro_calibration.py \
  --path_calib_dataset=/home/calibration_datasets/gopro-hero-12_polyumi_gripper_1 \
  --checker_size_m=0.021 \
  --image_downsample_factor=2 \
  --camera_model=FISHEYE \
  --recompute_corners=0 \
  --path_to_build build/applications/
```

## Docstring Formatting

This project enforces pydocstyle via ruff. The rules that come up most often:

- **D205** — multi-line docstrings require a blank line between the summary and the body:
  ```python
  # wrong
  """Summary line.
  More detail here.
  """
  # correct
  """Summary line.

  More detail here.
  """
  ```
- **D213** — the summary line of a multi-line docstring must start on the *second* line (after the opening `"""`):
  ```python
  # wrong
  """Summary line.

  Body.
  """
  # correct
  """
  Summary line.

  Body.
  """
  ```
- **D101/D102/D103** — public classes, methods, and functions need docstrings. One-line docstrings are fine for simple cases.

Run `uv run ruff check --fix .` to auto-fix the fixable ones, then address D205/D101 manually.
