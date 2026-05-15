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
This is a `uv` workspace. `ingest/` is the only workspace member. `pi/` is referenced as an editable path source (`tool.uv.sources`) so `polyumi_pi` is importable in the workspace venv, but it is not a member — it has its own `pi/.venv` managed separately for the Pi. Run `uv sync` at the root for PC-side dev dependencies. The `pi/` package requires `--system-site-packages` on the Pi for `picamera2`/`sounddevice`.

## Running Commands in the Right Environment

Always prefix Python and tool invocations with `uv run` from the repo root — never use bare `python`, `pip`, or `ruff`:

```bash
uv run ruff check .
uv run ruff format .
uv run python -c "import polyumi_ingest"   # ingest package
uv run pytest ...
```

`uv` selects the correct workspace venv automatically. Bare `python` / `pip` will pick up the wrong venv (e.g. `pi/.venv`) and produce "module not found" errors or install into the wrong place.

## Testing SLAM

The ORB-SLAM3 step (`OrbSlam3Step`, preprocessing step 2) reads its installation
path from environment variables. Set these before running any SLAM-related commands:

```bash
# We use the Cheng fork (slam/ORB_SLAM3_CHENG) — it has the fixes vanilla
# ORB-SLAM3 was missing (atlas-load activates the loaded map, null guards in
# LocalMapping, shutdown wait-for-threads, etc.). Our PolyUMI binaries
# (mono_inertial_gopro_vi_polyumi / mono_inertial_gopro_vi_localize) live in
# the same Examples/Monocular-Inertial dir alongside Cheng's stock binaries.
export ORB_SLAM3_DIR=/home/conor/Documents/W2026/winter_project/slam/ORB_SLAM3_CHENG
export ORB_SLAM3_BIN_SUBDIR=Examples/Monocular-Inertial
```

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
