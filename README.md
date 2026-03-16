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
