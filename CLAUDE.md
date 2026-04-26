# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolyUMI is a multimodal data collection system for robot imitation learning. A Raspberry Pi captures synchronized video (PiCamera2) and audio (sounddevice), either streaming live over ZMQ or recording episodes to disk. A host PC receives the stream via ROS2 nodes and visualizes in Foxglove Studio, or ingests recorded sessions into MP4s.

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

### Deploy to Pi
```bash
./deploy.sh <ssh_hostname>   # rsync pi/ + polyumi_pi_msgs to Pi, embeds git hash in _version.py
```

### Pi (run on device)
```bash
# Stream live data
python polyumi_pi/main.py stream

# Record an episode
python polyumi_pi/main.py record-episode --fps 10 --robot polyumi_gripper --task <task_name>
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
cd ingest
python main.py fetch --host <hostname> --latest
python main.py process-video recordings/session_YYYY-MM-DD_hh-mm-ss
python main.py process-all --force
```

## Architecture

There are two deployment contexts with different data flows:

### Arm End Effector (EE)
Pi and GoPro are independent. Pi streams its camera and audio live over ZMQ; GoPro streams independently over HDMI into a capture card on the host PC.
```
Raspberry Pi
  CameraStreamer (PiCamera2)  ──┐
  AudioStreamer (sounddevice) ──┼─→ ZMQ PUSH (tcp://*:5555/5556)
                                         ↓
                                pi_receiver_node.py (ROS2)
                                → publishes CompressedImage + RawAudio
                                → foxglove_bridge (WebSocket :8765)
                                → Foxglove Studio

GoPro ──→ HDMI ──→ Capture Card ──→ host PC (separate pipeline)
```

### Gripper (not fully implemented)
No streaming. Pi writes its camera and audio to local files. GoPro writes video to its SD card. The Pi interacts with the GoPro only to:
1. Synchronize timestamps between the two systems
2. Trigger GoPro recording in response to a button press handled by the Pi

### Key Modules

- **`pi/polyumi_pi/main.py`** — Typer CLI; entry point for all Pi operations
- **`pi/polyumi_pi/cam_streamer.py`** / **`audio_streamer.py`** — run in separate processes; communicate stats back via `multiprocessing.Pipe`
- **`pi/polyumi_pi/files/`** — `SessionFiles` manages `metadata.json`, JPEG frame storage, and WAV audio; `session.py` is the main facade
- **`pi/polyumi_pi/gopro/`** — GoPro integration via open-gopro SDK
- **`ros2_ws/src/polyumi_pi_msgs/`** — Protobuf definitions (`CameraFrame`, `AudioChunk`) with nanosecond timestamps; generated `*_pb2.py` files live alongside `.proto` sources
- **`ros2_ws/src/polyumi_ros2/`** — ROS2 package; `pi_receiver_node.py` bridges ZMQ → ROS2 topics
- **`ingest/main.py`** — fetches sessions from Pi via tar-over-SSH, then encodes JPEG frames + WAV → MP4 via ffmpeg

### Session Data Layout
```
~/recordings/session_YYYY-MM-DD_hh-mm-ss/
├── metadata.json
├── video/frame_000001.jpg ...
└── audio.wav
```

### Dependencies
- **Pi**: `pyzmq`, `protobuf`, `picamera2`, `sounddevice`, `open-gopro`, `opencv-python-headless`, `rpi-hardware-pwm`, `typer`, `numpy`
- **ROS2**: `rclpy`, `foxglove_bridge`, `sensor_msgs`, `foxglove_msgs`
- **Ingest**: same as Pi + external `ffmpeg` binary
- **Dev**: `pytest`, `ruff`

### Package Management
This is a `uv` workspace. Root `pyproject.toml` declares workspaces `pi/` and `ingest/`. Run `uv sync` at the root for PC-side dev dependencies. The `pi/` package requires `--system-site-packages` on the Pi for `picamera2`/`sounddevice`.
