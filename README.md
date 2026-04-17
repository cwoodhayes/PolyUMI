https://github.com/user-attachments/assets/2f902600-9682-4e67-a75c-fc8fa358cb92

# PolyUMI: Visual + Auditory + Tactile Manipulation Platform for Imitation Learning

**Project website:** https://cwoodhayes.github.io/projects/polyumi

PolyUMI is a real-time data collection & control platform for robotic imitation learning, which unifies the following sensor modalities in a single end-effector:
- **touch** (via a custom optical tactile-sensing finger, based off of [PolyTouch](https://polytouch.alanz.info/)) - *10fps 540x480 MJPEG video (MP4)*
- **mechanical vibration** (via a contact microphone fixed to the finger housing) - *16kHz PCM audio (WAV)*
- **vision** (via GoPro camera on wrist + finger camera peripheral vision) - *60fps 1920x1080 MJPEG video (MP4) + 10fps 540x480 MJPEG video*
- **proprioception** (via monocular inertial SLAM from GoPro + IMU in gripper, or robot joint encoders + FK in embodiments)

It combines the [Universal Manipulation Interface (UMI)](https://umi-gripper.github.io/) platform with a custom touch-sensing finger inspired by the [PolyTouch tactile + audio sensor](https://polytouch.alanz.info/), with hardware, firmware, and software built from scratch for a modern robotics stack (ROS2 Kilted + Python 3.13 + Foxglove visualizer).

<div align="center" style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;">
  <div style="flex: 1 1 480px; min-width: 320px; max-width: 600px;">
    <img src="docs/dataflow_overview.png" alt="Dataflow Overview" style="width: 100%;"/>
    <p style="margin: 6px 0 0; font-size: 0.9em; color: #666;">Data flow through the PolyUMI system.</p>
  </div>
  <div style="flex: 1 1 480px; min-width: 320px; max-width: 600px;">
    <img src="docs/polyumi sw components.png" alt="Software Components" style="width: 100%;"/>
    <p style="margin: 6px 0 0; font-size: 0.9em; color: #666;">General summary of software components in this repo.</p>
  </div>
</div>

## Repo Structure

```
pi/               # RPi client: camera, audio, LED streaming + episode recording
postprocess/      # PC-side CLI: fetch sessions from Pi, encode video
ros2_ws/
  src/
    polyumi_pi_msgs/   # Protobuf message definitions (camera frame, audio chunk)
    polyumi_ros2/      # ROS 2 nodes + Foxglove launch files
```

## Prerequisites

**PC:** Python 3.13, [uv](https://github.com/astral-sh/uv), ROS 2 Kilted, `ffmpeg`, `protobuf-compiler`

**RPi:** Raspberry Pi Zero 2W flashed with Raspberry Pi OS. See [Hardware Notes](#hardware-notes) for HAT-specific config.

## Installation

### PC

Install postprocessing dependencies (includes the `polyumi_pi` package for shared data types):

```bash
uv sync --group dev
```

Build the ROS 2 workspace:

```bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r --rosdistro kilted
colcon build
source install/setup.bash
```

### RPi

Deploy code to the Pi (run from repo root on your PC). This also stamps the current git commit hash into the Pi package:

```bash
./deploy.sh <pi_ssh_hostname>
```

Then on the Pi, install the Python environment:

```bash
cd ~/pi
uv venv --system-site-packages
uv sync --no-dev
uv pip install -e ~/polyumi_pi_msgs
uv pip install -e .
```

`picamera2` must be installed via `apt`, not pip — the `--system-site-packages` flag above pulls it in from the system.

**Tip for development:** add the `deploy.sh` invocation to `.vscode/tasks.json` as a build task so it runs on every save.

## Recording Data

On the Pi, from the `~/pi` directory:

```bash
python polyumi_pi/main.py record-episode
```

This writes a timestamped `session_YYYY-MM-DD_HH-MM-SS/` directory to `~/recordings/` containing:
- `video/` — JPEG frames with a `video_timestamps.csv` sidecar
- `audio.wav` — 16-bit PCM
- `metadata.json` — session config, frame counts, drop stats, git version

Stop recording with `Ctrl+C`. Metadata is finalized on exit.

## Postprocessing

From the repo root:

```bash
cd postprocess
```

**Fetch the latest session from the Pi:**

```bash
python main.py fetch --host <pi_ssh_hostname> --latest
```

**Fetch all new sessions:**

```bash
python main.py fetch --host <pi_ssh_hostname>
```

Transfer uses tar-over-ssh (faster than rsync for many small JPEG files). Add `--verbose-transfer` to debug transfer issues.

**Encode a single session to MP4** (audio is muxed in automatically if present):

```bash
python main.py process-video recordings/session_YYYY-MM-DD_HH-MM-SS
```

**Encode all unprocessed sessions:**

```bash
python main.py process-all
```

Options: `--force` to re-encode existing outputs, `--no-include-audio` to skip audio mux, `--output-name custom.mp4` to change the output filename.

## Streaming / Demos

### Streaming Demo

Streams camera, audio, and GoPro wrist camera into Foxglove.

On the Pi:

```bash
python polyumi_pi/main.py stream
```

On the PC:

```bash
ros2 launch polyumi_ros2 stream_demo.launch.xml
```

Open [Foxglove](https://app.foxglove.dev), connect to `ws://localhost:8765`, and drag in `ros2_ws/src/polyumi_ros2/foxglove/layouts/stream_demo.json`.

The launch file accepts two arguments: `pi_host` (default `10.106.10.62`) and `video_device` (default `/dev/video2`) for the GoPro capture device.

### Franka Demo

Same as above but also launches MoveIt with a Franka arm URDF visualization.

On the PC:

```bash
ros2 launch polyumi_ros2 franka_demo.launch.xml
```

Use the `franka_demo.json` Foxglove layout. Requires `franka_fer_moveit_config` — see that repo for setup.

## Hardware Notes

### Audio HAT (Waveshare WM8960)

The default RaspiAudio driver conflicts with the hardware PWM used for the LED. Instead, use the Waveshare DKMS driver:

```bash
git clone https://github.com/waveshare/WM8960-Audio-HAT
cd WM8960-Audio-HAT
sudo ./install.sh
sudo reboot
```

In `/boot/firmware/config.txt`, move PWM away from GPIO18/19 (which are used by I2S) to GPIO12/13:

```
dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4
```

Validate audio capture works before proceeding:

```bash
arecord -D hw:wm8960soundcard -r 48000 -f S16_LE -c 2 -d 5 test.wav
```

### LED Circuit

The LED strip is driven by an AO3400A N-channel MOSFET via hardware PWM channel 0 on GPIO12 (header pin 32). The PWM overlay above is required for this to work alongside the audio HAT. See [my portfolio post](https://cwoodhayes.github.io/projects/polyumi) for more details until I can write up hardware docs here.

### PiSugar Battery

Battery status is accessible at `http://<pi_ip>:8421` or via I2C:

```bash
i2cdetect -y 1
i2cget -y 0x57 0x2a   # battery percentage
```

## Troubleshooting

**`_version.py` missing on the Pi** — run `./deploy.sh <pi_ssh_hostname>` from the PC; this generates the file from the current git HEAD.

**Audio not detected** — confirm `wm8960-soundcard` appears in `arecord -l`. If the default RaspiAudio driver was previously installed, the Waveshare DKMS driver may need to be reinstalled after a kernel update.

**Wi-Fi not listing on the Pi** — run `sudo modprobe brcmfmac`, then retry `nmcli dev wifi connect "your-network"`.

**ZMQ frames dropping** — check the `cb_drops` counter in the Pi logs. The audio streamer uses a 100-frame queue with drop-and-replace on overflow; the video streamer uses `NOBLOCK` sends with a high-watermark of 2. Persistent drops indicate the network link is the bottleneck.

**`protoc` not found during `polyumi_pi_msgs` install** — install `protobuf-compiler` (`sudo apt install protobuf-compiler` on the Pi, or via your system package manager on the PC).
