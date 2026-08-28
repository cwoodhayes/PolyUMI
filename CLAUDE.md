# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolyUMI is a multimodal data collection system for robot imitation learning. See [README.md](README.md) for a full description, architecture diagrams, and usage instructions.

## Common Commands

### Linting

Fix everything fixable, in one command:
```bash
uv run ruff check --fix . && uv run ruff format .
```
Fix first, format last, so the formatter always has the last word over any autofix.
To check without writing (what CI runs): `uv run ruff check . && uv run ruff format --check .`

**What that does and does not fix.** Formatting is fully automatic — whitespace, quotes, line
length, trailing commas never need hands. Lint mostly is not: this repo selects `E`, `F`, `D`,
`W`, and the findings are dominated by `D` (pydocstyle), which is unfixable by construction —
ruff cannot invent a docstring. In the b6cfe76 cleanup `ruff format` fixed 20 of 20 files
unaided, while `--fix` resolved 1 of 13 lint errors; the other 12 were missing docstrings and
two hand edits. So expect the command to leave `D1xx` behind for you to write.

Two traps worth knowing:
- `D1xx` only fires on *public* modules — a file named `_foo.py` is private, so nothing in it
  is ever reported missing a docstring.
- `--unsafe-fixes` will clear things like `F841` (unused variable) by deleting the assignment.
  That drops the right-hand side too, so it changes behaviour when the RHS has side effects.
  It is correctly off by default; reach for it per-file, not repo-wide.

ruff is pinned (`[dependency-groups] lint` in pyproject.toml) because its formatter output
changes between minor releases — an unpinned ruff lets two developers reformat the same files
back and forth. Use `uv run ruff`, not `uvx ruff`, so you get the pinned version.
`.github/workflows/lint.yml` enforces this on every PR; it installs only the `lint`
group, since the `dev` group's picamera2 is Pi-only and will not build on a runner.
`external/` is excluded — those are third-party submodules carrying upstream style.

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

**ROS2 tests** need ROS's interpreter (they import `rclpy`), so `uv run` is wrong here — use
`/usr/bin/python3` with the workspace sourced and `VIRTUAL_ENV` unset:
```bash
bash -c 'unset VIRTUAL_ENV; cd ros2_ws && source /opt/ros/kilted/setup.bash \
  && source install/setup.bash && cd src/polyumi_ros2 \
  && /usr/bin/python3 -m pytest test/test_policy_client_node.py -q'
```
`colcon test --packages-select polyumi_ros2` also runs them, and is expected to pass clean:
```bash
bash -c 'unset VIRTUAL_ENV; cd ros2_ws && source /opt/ros/kilted/setup.bash \
  && colcon test --packages-select polyumi_ros2 \
  && colcon test-result --test-result-base build/polyumi_ros2'
```
The **NUC bridges** in `nuc/` are standalone scripts, not an ament package, so `colcon test`
never sees them. `nuc/test_*.py` run on the laptop anyway — they need only the `franka_msgs` /
`moveit_msgs` message definitions (built in `ros2_ws`, or from `/opt/ros`) and mock the service
and action clients, so no hardware, no move_group, and no NUC is involved:
```bash
bash -c 'unset VIRTUAL_ENV; source /opt/ros/kilted/setup.bash \
  && source ros2_ws/install/setup.bash \
  && /usr/bin/python3 -m pytest nuc/ -q'
```

The generated `ament_copyright` / `ament_flake8` / `ament_pep257` linter tests were **deleted** —
their ROS defaults (99 cols, ament import order) contradicted this repo's ruff config, so
`colcon test` failed regardless of whether real tests passed. Python style is ruff's job alone;
`ruff check ros2_ws/` is expected to be clean. Only `ament_xmllint` remains, since nothing else
validates `package.xml`. Don't re-add the others when generating new ROS packages.

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
On the NUC, two launch files: `nuc/launch/fr3_bringup.launch.py` (the hardware session —
franka_bringup + the `fr3_arm_controller` spawner, replacing the `fr3-bringup` +
`fr3-arm-controller` alias pair) and `nuc/launch/fr3_inference.launch.py` (move_group + both
PolyUMI bridges, with separate `execute_arm` / `execute_gripper` flags, both defaulting false).
They are deliberately two files: bringup is the crash-prone, FCI-gated piece and must be
restartable on its own.

**The TCP frame is `polyumi_tcp`, not `fr3_hand_tcp`.** The policy's poses live on the
closed-fingertip midpoint in GoPro-optical axes (ingest step 5's `hand` body frame), so both the
observation TF lookup (`eef_frame`) and the MoveIt bridge's target link name that frame. Its
transform is defined **once**, in `nuc/tcp_calib.py`, and reaches its two consumers from there:
TF via a `static_transform_publisher` in `fr3_bringup.launch.py`, and move_group's RobotModel via
`nuc/description/fr3_polyumi.urdf.xacro` (a thin wrapper over `fr3.urdf.xacro`) which
`fr3_move_group.launch.py` feeds it. Never hardcode the numbers a second time. Note
`franka_msgs/srv/SetTCPFrame` (libfranka `setEE`) is *not* the lever — it only changes `O_T_EE`
reporting, while TF and MoveIt are driven entirely by the URDF. The transform is measured from
CAD (the same method UMI uses), not from a calibration rig; what remains unvalidated is the
real GoPro mount tilt — the geometry itself, `Rz(+90°)` sign included, is confirmed on hardware by
`ros2 run polyumi_ros2 tcp_pivot_test`, which pivots about the TCP so you can watch whether the
closed fingertips hold still. Re-run it after any change to the mount, the fingers, or this file.

**The end-effector payload lives in `nuc/tcp_calib.py` too** (`PAYLOAD_MASS`, `PAYLOAD_COM_HAND`),
pushed to the FCI once by `fr3_bringup.launch.py` via `franka_msgs/srv/SetLoad`, ordered ahead of
the `fr3_arm_controller` spawner so no control loop is running when it lands. An under-configured
payload shows up as the TCP dropping the instant the impedance controller activates and holding a
steady-state offset — `Δz = m_unmodelled · g / K_trans`, so at K=2000 N/m 1 mm of droop is 0.2 kg.
Note the CoM is written in `fr3_hand` (the frame `TCP_XYZ` uses) and converted to the flange by
`payload_com_flange()`; the URDF is *not* a lever here, since `franka_hardware` reads no payload
out of it. Two FR3 constraints make `SetLoad` fail in ways the service response hides behind the
string `"command exception error"` — a nonzero mass needs a nonzero inertia tensor
(`payload_inertia_flange()` derives one), and the call is refused entirely while any controller holds the
arm (`current mode ("Move")`), which is why bringup sequences it ahead of the spawner. **A failed
`SetLoad` aborts bringup** — `nuc/set_payload.py` reads `response.success` off the typed message
rather than the CLI's exit status, which is 0 either way, because the
spawner immediately makes it unretryable and a wrong gravity model is easy to miss. The real
message is only in the `/service_server` log. Full procedure in
[docs/calibration-instructions.md](docs/calibration-instructions.md).

**`./fr3_session.sh`** (repo root) builds the whole wall — NUC, Pi, GPU box, laptop — as one
tmux session, running the safe commands and pre-typing the robot-moving ones for you to
confirm. Every fresh start (not a re-attach) also rsyncs `nuc/` to the NUC and runs
`./deploy.sh` for the Pi, so both run this working copy rather than whatever they last had —
`SKIP_DEPLOY=1` skips that for a faster re-launch. Re-run to re-attach after a disconnect; the
NUC/GPU-box panes are remote tmux, so they survive. Full reference and the exact
environment assumptions live in [docs/crb-fr3-inference.md](docs/crb-fr3-inference.md).

**Clock sync (this setup):** the NUC and laptop must agree on wall time or TF lookups fail
with "extrapolation into the past". The NUC (`jailfranka`) is on a jailed VLAN that blocks
outbound **UDP 123**, so it cannot reach public NTP — it syncs to the **laptop over the
`10.0.0.x` arm link** instead. This is wired durably via chrony drop-ins (persist across
reboots): the NUC has `/etc/chrony/conf.d/laptop-time.conf` = `server 10.0.0.1 iburst prefer`,
and the laptop has `/etc/chrony/conf.d/allow-fr3-link.conf` = `allow 10.0.0.0/24` (plus its
existing internet pools + `local stratum 5`). Verify with `ssh jailfranka chronyc sources` →
expect `^* 10.0.0.1` (sub-ms offset). If it ever drifts again, the laptop was probably down /
off `10.0.0.1` when the NUC booted (the NUC then falls back to its own `local stratum 10`
clock); `ssh jailfranka 'sudo chronyc makestep'` once the link is up re-steps it. With this in
place, `tf_use_latest` is no longer needed for real runs — it was only a stationary-dry-run
crutch for the old skew.

**When debugging FR3 inference on the arm — read [docs/crb-fr3-inference.md](docs/crb-fr3-inference.md)
FIRST, especially "When it doesn't come up" and "Gripper problems", before re-diagnosing.** The common failure modes and
their fixes are documented there: nothing publishing / Foxglove blank (a duplicate or leftover
launch grabbing port 8765 + `/dev/video2` — `pkill` leftovers and confirm a single stack); TF
"extrapolation into the past" (laptop↔NUC clock skew — should be fixed durably by the chrony
setup below; if it recurs, re-step the NUC clock, don't reach for `tf_use_latest`); TF "fr3_link0 does not exist" (NUC `fr3-bringup` crashed — restart it);
"capture pipeline stalled" (the Elgato's ~200 ms 1080p convert latency — `max_image_age_s:=0.3`).
The **dry-run** pattern (validate commanded motion without moving the arm) is `execute_motion:=false`
(default) + watch `/polyumi/target_poses_preview` in Foxglove.

### Ingest (host PC)
```bash
pingest --help
pingest fetch --host <hostname> --latest
pingest process-all --force
pingest export <scene> -o <name>.zarr.zip                       # visuomotor dataset
pingest export <scene> -o <name>.zarr.zip --type polyumi        # + data/mic_0 (contact mic, needs pp 6)
                                                                #   and data/finger_rgb (finger camera, cropped)
```

### Training the diffusion policy (GPU workstation, Docker)
Training runs the UMI fork (`external/polyumi_diffusion_policy`) in a Docker image built from
its conda env via micromamba — **not** bare conda (which fights ROS) and **not** the uv
workspace. One image serves both training and inference. Run it with `./train_policy.sh`
(builds the fork image + mounts dataset/output with rootless-safe flags). Full walkthrough,
including rootless-Docker gotchas, is in
[docs/training-instructions.md](docs/training-instructions.md). This is the step after `pingest
export`.

**Serving a trained checkpoint** (the real inference server) uses the same image via
`CKPT=/abs/path/to/<name>.ckpt ./serve_policy.sh` at the repo root (the inference counterpart of
`train_policy.sh`). It runs `serve_policy.py` in-container and direct-imports the policy — there is
**no** subprocess/`conda run`. Do **not** run the fork's `external/polyumi_diffusion_policy/docker/serve.sh`
on the host; it is the in-container entrypoint and fails with `exec: uvicorn: not found`. The wire
contract matches `dummy_server` (`POST /predict_cartesian/` + `POST /reset` for the episode-start
pose); the ROS-side `policy_client_node` derives the `/reset` URL from the predict URL. Serving and
training must use the same image because checkpoints are dill-pickled against the exact dep tree.

## Key Modules

- **`pi/polyumi_pi/main.py`** — Typer CLI; entry point for all Pi operations (`polyumi-pi`)
- **`pi/polyumi_pi/cam_streamer.py`** / **`audio_streamer.py`** — run in separate processes; communicate stats back via `multiprocessing.Pipe`
- **`pi/polyumi_pi/files/session.py`** — `SessionFiles` manages `metadata.json`, JPEG frame storage, and WAV audio
- **`pi/polyumi_pi/files/scene.py`** — `SceneFiles` groups one or more sessions under a shared scene directory
- **`pi/polyumi_pi/gopro/`** — GoPro integration via open-gopro SDK
- **`ros2_ws/src/polyumi_pi_msgs/`** — Protobuf definitions (`CameraFrame`, `AudioChunk`) with nanosecond timestamps; generated `*_pb2.py` files live alongside `.proto` sources
- **`ros2_ws/src/polyumi_ros2/`** — ROS2 package; `pi_receiver_node.py` bridges ZMQ → ROS2 topics
- **`ingest/polyumi_ingest/main.py`** — `pingest` CLI; fetches sessions from Pi via tar-over-SSH, builds pzarr working-format stores, and archives scenes to zip
- **`catalog/polyumi_catalog/`** — `polyumi-catalog` CLI (`sync`, `serve`); FastAPI + HTMX browser over a SQLite cache of the recordings tree

## The Catalog DB Has No Migrations — Ever

`catalog.db` is a **pure cache**. The recordings tree (`metadata.json`, `scene.json`, the
pzarr stores) and the dataset manifests are the source of truth; every row is re-derivable by
`polyumi-catalog sync`, and rebuilding costs seconds. So:

- **Never write a migration, an `ALTER TABLE`, or a schema-version column.** To add or change a
  field, just edit `catalog/polyumi_catalog/models.py` and make sure `sync.py` populates it.
- Startup compares the DB's schema against the models (`db.schema_mismatches`). On any
  difference the CLI prints what drifted and offers to rebuild (`db.rebuild_schema` →
  drop + create + re-sync). `--rebuild-db` / `--no-rebuild-db` answers it non-interactively.
- Anything the catalog caches that is *not* re-derivable from disk is a bug in that field's
  design, not a reason to preserve the DB file.

## Session Data Layout
```
~/recordings/
└── scene_YYYY-MM-DD_hh-mm-ss_XXXX/
    └── session_YYYY-MM-DD_hh-mm-ss_XXXX/
        ├── metadata.json
        ├── video/frame_000001.jpg ...
        └── audio.wav
```

## Package Management
This is a `uv` workspace. `ingest/` is the only workspace member. `pi/` is referenced as an editable path source (`tool.uv.sources`) so `polyumi_pi` is importable in the workspace venv, but it is not a member — it has its own `pi/.venv` managed separately for the Pi. `inference_server/` is also deliberately **not** a member: it must import under three interpreters (see below), so it keeps its own `inference_server/.venv` — run the dummy server with `cd inference_server && uv run dummy-server`. Run `uv sync` at the root for PC-side dev dependencies. The `pi/` package requires `--system-site-packages` on the Pi for `picamera2`/`sounddevice`.

## The Inference Protocol Lives in One Library

`inference_server/` is the `polyumi_inference` package: the observation wire format, the client
that speaks it, and the FastAPI app that answers. **Both ends import it.** The ROS-side
`policy_client_node` holds a `PolicyClient`; `dummy_server` and the fork's `serve_policy.py` are
`PolicyBackend`s behind `create_app`. A server writes no routes and no validation — that is what
makes "the dummy refuses exactly what a checkpoint refuses" true by construction. It used to be
three byte-identical copies of one file and a test that compared them.

Layers, innermost out: `wire.py` (bytes ↔ arrays) → `types.py` (`Observation`, `ActionChunk`) →
`contract.py` (what the policy requires) → `client.py` / `server.py`.

**It must stay Python 3.9 compatible.** The policy container's conda env is `python=3.9` /
numpy 1.24 while the ROS node is 3.12, and both import this. Every module carries
`from __future__ import annotations`; `test_python39_floor.py` asserts that, since a `X | None`
annotation raises at import on 3.9 and is invisible on 3.12. Verify a real change against the
floor by building the image and importing there — that is the only place it is actually exercised.

**Installing it for the ROS node** (ament_python under `/usr/bin/python3`, PEP 668):

```bash
pip install --user --break-system-packages --no-deps -e inference_server/
```

`--no-deps` because numpy and requests come from apt via rosdep; letting pip resolve them would
shadow the system numpy the rest of the ROS stack links against. Editable, so it tracks the
working copy.

**Getting it into the policy container** is a two-stage `docker build` (`build_policy_image.sh`,
used by both `train_policy.sh` and `serve_policy.sh` so the two roles run the same image). The
fork's Dockerfile builds with the *fork directory* as its context and cannot see
`inference_server/`, so `docker/polyumi_inference.Dockerfile` layers the library on top with
`inference_server/` as its own context. Do **not** "fix" this by staging a copy into the fork —
that re-creates the duplicated file this library exists to delete.

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

If a `rosidl` build (e.g. `franka_msgs`) fails with `ModuleNotFoundError: No
module named 'em'`, that's the same `VIRTUAL_ENV=pi/.venv` problem: the build is
using the Pi venv's `python3`, which lacks `empy`. The system `/usr/bin/python3`
has it (`python3-empy`). Unset `VIRTUAL_ENV` *inside* the `bash -c` (not just
before it) so it doesn't get re-inherited:
`bash -c 'unset VIRTUAL_ENV; cd ros2_ws && source /opt/ros/kilted/setup.bash && colcon build ...'`.

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

## Comment Content

Comments explain the code as it is now. **Git history is where the code's past lives** — a
comment that narrates what the code used to be, when it changed, or which incident forced the
change is duplicating `git log`, and unlike `git log` it goes stale silently.

Cut on sight:

- **"This used to …" / "Deliberately omitted until …" / "X was tried and failed"** — the
  rejected alternative belongs in the commit message. Keep it only when a reader would
  otherwise re-introduce it, and then in one clause, not a paragraph.
- **Dates on changes.** "Fixed 2026-08-18", "seen on 2026-08-17". Dates on *measurements* are
  the exception and should stay: a calibrated constant needs to say when it was measured
  (`gripper_calib.yaml`, `tcp_calib.py`).
- **Incident forensics.** Which scene broke, how many episodes it cost, the exact log line.
  State the failure mode in the present tense instead; the postmortem goes in the commit.
- **Personal anecdote.** "mine sets 63", "cost an hour".

Keep, always: why the code is shaped this way, constraints that still bind, where a magic
number came from, upstream behaviour we are matching, and gotchas a reader would trip on.

Two more rules of thumb:

- **Don't duplicate `docs/`.** For anything with a doc section (the FR3 troubleshooting modes
  in `docs/crb-fr3-inference.md`, especially), state the mechanism in a sentence or two and
  point at the doc. Two copies of a narrative drift.
- **If the explanation is longer than the code it explains, it is probably a doc, a commit
  message, or a docstring — not an inline comment.**

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
