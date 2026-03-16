# PolyUMI - Visual-Auditory-Tactile Manipulator for Imitation Learning

Author: Conor Hayes

## Setup

### PC setup
ROS2 environment setup (for inference/streaming)
```bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r --rosdistro kilted
colcon build
source install/setup.bash
```

Postprocessing setup (for after recording on umi)
```bash
# at repo root
uv sync --group dev
```

### RPi setup

#### System setup
TODO
- flash the image as configured by me
- install various things

```bash
sudo apt install \
    protobuf-compiler
```

#### Library setup
Run on your PC:
```bash
# copy essential libraries to the pi
PI_USER="your pi's user here"
PI_ADDR="your pi's IP address here"
rsync -av --delete --exclude='.venv/' pi $PI_USER@$PI_ADDR:~
rsync -av --delete --exclude='.venv/' ros2_ws/src/polyumi_pi_msgs $PI_USER@$PI_ADDR:~",
```

Run on the pi:
```bash
cd pi
uv venv --system-site-packages
uv sync --no-dev
uv pip install -e ~/polyumi_pi_msgs
uv pip install -e .
```

**Recommended for Development**: if using VS Code, add the `rsync` commands above to your `.vscode/tasks.json` as a build command.

## Postprocess Workflow (Fetch -> Process)

After recording sessions on the Pi, use the `postprocess` CLI on your PC to copy and convert data.

From the repo root:

```bash
cd postprocess
```

### 0) Record data on the Pi

On the Pi, from the `pi` directory:

```bash
python main.py record-episode
```

This writes a new `session_*` directory under `~/recordings` on the Pi.

### 1) Fetch sessions from the Pi

Fetch latest session only:

```bash
python main.py fetch --host [your hostname] --latest
```

Fetch all sessions not already present locally:

```bash
python main.py fetch --host [your hostname]
```

Notes:
- Session discovery still uses `ssh + ls` first, so you get a count before copy starts.
- Transfer uses tar-over-ssh (faster for many small frame files).
- Use `--verbose-transfer` if you want detailed transfer output for debugging.

### 2) Process video for one session

```bash
python main.py process-video recordings/session_YYYY-MM-DD_hh-mm-ss
```

This creates `finger.mp4` in that session directory, and includes `audio.wav` when available.

### 3) Process all unprocessed sessions

```bash
python main.py process-all
```

This scans `recordings/session_*`, skips sessions that already have `finger.mp4`, and processes the rest.

Useful options:
- Re-encode everything: `python main.py process-all --force`
- Change output name: `python main.py process-all --output-name custom.mp4`
- Disable audio mux: `python main.py process-all --no-include-audio`

## Run Demos

### Streaming Demo
This demo streams data from all sensors simultaneously into Foxglove.

On the RPi, from the `pi` directory after the setup steps in [Library Setup](#library-setup) above:
```bash
python polyumi_pi/main.py stream
```

On PC:
```bash
# launch the demo
ros2 launch polyumi_ros2 stream_demo.launch.xml
```
Then open [foxglove](https://app.foxglove.dev) in your browser, and connect to `ws://localhost:8765` (the default).
Drag and drop `ros2_ws/src/polyumi_ros2/foxglove/stream_demo.json` into the UI.

### Franka Demo
This demo is the streaming demo for the PolyUMI Franka end-effector, which includes a visualization of the real-time movements of the Franka arm. Must be connected to the arm, of course.

TODO - explain how to get matt's franka repo

On the RPi, from the `pi` directory after the setup steps in [Library Setup](#library-setup) above:
```bash
python polyumi_pi/main.py stream
```

On PC:
```bash
# launch the demo
ros2 launch polyumi_ros2 franka_demo.launch.xml
```
Then open [foxglove](https://app.foxglove.dev) in your browser, and connect to `ws://localhost:8765` (the default).
Drag and drop `ros2_ws/src/polyumi_ros2/foxglove/franka_demo.json` into the UI.
