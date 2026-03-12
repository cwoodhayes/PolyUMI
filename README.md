# {Repo Name}

Author: Conor Hayes

## Setup

### PC setup
```bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r --rosdistro kilted
colcon build
source install/setup.bash
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
```

**Recommended for Development**: if using VS Code, add the `rsync` commands above to your `.vscode/tasks.json` as a build command.

## Run Franka Streaming Demo
```bash
# launch the demo
ros2 launch polyumi_ros2 stream_demo.launch.xml
```
Then open [foxglove](https://app.foxglove.dev) in your browser, and connect to `ws://localhost:8765` (the default).