# CRB FR3 Inference Setup

**Note for Northwestern CRB members**: This document describes how to run inference on the CRB lab's Franka FR3 arm (in the student office, connected to the NUC with the skull on it). 

**Note for users outside of Northwestern**: This is specific to our equipment, but is likely still useful as an example to bring up inference on your own equipment. This documents the specific two-machine
setup used in the CRB lab to drive a **Franka FR3** for PolyUMI inference. The
distro split, IP plan, DDS choice, and NUC aliases below are particular to this
hardware — adapt them for your own robot and network rather than copying verbatim.
For the lab-agnostic inference architecture and API contract, see
[franka-inference-bringup.md](franka-inference-bringup.md).

It captures the laptop and NUC environments and the DDS contract that lets a Kilted
laptop talk to a Humble NUC. If something here drifts from reality, fix it here —
`setup_franka_env.sh` and `ros2_ws/config/cyclonedds_laptop.xml` assume these values.

## Setup
### Topology

```
Laptop (Kilted, Noble)                        NUC (Humble, Jammy)  [nu-crb]
RMW=rmw_cyclonedds_cpp, DOMAIN=0  ◄─ DDS over ─►  RMW=rmw_cyclonedds_cpp, DOMAIN=0
enp0s31f6 = 10.0.0.1/24            10.0.0.x      enx00249b860356 = 10.0.0.2/24
  - foxglove_bridge                               - fr3_bringup.launch.py (franka_bringup
  - v4l2_camera (GoPro)                             + fr3_arm_controller spawner)
  - pi_receiver_node                              - move_group  (nuc/launch/fr3_move_group.launch.py)
  - policy_client_node ──HTTP──┐                  - fr3_moveit_bridge  (nuc/fr3_moveit_bridge.py)
  - dummy_server (localhost:8000) ◄┘              - fr3_gripper_bridge (nuc/fr3_gripper_bridge.py)
        │                                         - publishes fr3_* TF + joint states
        │                                         - enp89s0 = 192.168.51.10 → robot @ .20
        ├── /polyumi/target_poses (PoseArray) ─────────► fr3_moveit_bridge ──► move_group
        └── /polyumi/target_gripper ───────────────────► fr3_gripper_bridge ─► /fr3_gripper/{move,grasp}
              (JointTrajectory)
```

The PolyUMI ROS2 nodes use only distro-agnostic APIs (`rclpy`, `sensor_msgs`,
`tf2_ros`, `foxglove_msgs`), so they run on the laptop under Kilted. The Franka
stack is Humble-only and stays on the NUC; the two machines interoperate purely at
the DDS wire level.

**Motion execution is split deliberately.** The laptop does *not* call MoveIt — it
publishes the returned inference **action chunk** as a `geometry_msgs/PoseArray` on
`/polyumi/target_poses` (one waypoint per action step, `n_action_steps` long — see
[chunk execution](#action-chunk-execution) below), and the NUC-side `fr3_moveit_bridge`
does all the MoveIt calls against its **local** move_group, planning the whole chunk as
one multi-waypoint Cartesian path. This is not a style choice: large nested MoveIt
action goals get corrupted across the laptop/NUC rmw-version gap (see
[rmw mismatch](#rmw-version-mismatch--what-is-and-isnt-harmless)). Small messages like
`PoseArray` cross fine, so the pose chunk is the interop boundary.

### Action-chunk execution

`policy_client_node` requests an `n_action_steps`-long chunk from the inference server
each tick (param `n_action_steps`, default **8**) and publishes the *whole chunk* as one
`PoseArray`, rather than just `actions[0]`. This is deliberate, not incidental: at
10 Hz a single-waypoint target is a discrete ~2 cm hop the arm cannot track in real
time, and `fr3_moveit_bridge`'s skip-while-busy would drop nearly every tick. Publishing
the full chunk lets `move_group` plan one smooth multi-waypoint Cartesian path per chunk
(receding-horizon control, the standard UMI/DP execution pattern) instead of stuttering
between unreachable single-step goals. The bridge still applies skip-while-busy at the
*chunk* level — if a chunk is still executing when the next one is published, the new
one is dropped and picked up on the next available tick.

### User PC (i.e. my personal Ubuntu laptop)

| | |
|---|---|
| OS | Ubuntu 24.04 Noble |
| ROS2 | Kilted |
| Wired NIC | `enp0s31f6`, static **`10.0.0.1/24`** via NM profile `fr3-link`, direct cable to the NUC's `enx` |
| RMW | `rmw_cyclonedds_cpp` — `sudo apt install ros-kilted-rmw-cyclonedds-cpp` |
| `ROS_DOMAIN_ID` | `0` |
| `CYCLONEDDS_URI` | `ros2_ws/config/cyclonedds_laptop.xml` |
| `franka_msgs` | built from the `external/franka_ros2` submodule (see below) |
| Env | `source setup_franka_env.sh` (repo root) sets all of the above |

**`franka_msgs` (FR3 custom message/service types).** The NUC publishes
`franka_msgs/msg/FrankaRobotState` and `franka_msgs/srv/*`, which we need to build from source in
the `frankarobotics/franka_ros2` submodule (pinned to **`v0.1.15`**, matching the
NUC). It's `rosidl`-only — no libfranka — so it builds cleanly on Kilted:

```bash
git submodule update --init external/franka_ros2     # after a fresh clone
# ros2_ws/src/franka_msgs is a symlink into the submodule; build just that package:
unset VIRTUAL_ENV; bash -c 'cd ros2_ws && source /opt/ros/kilted/setup.bash && colcon build --packages-select franka_msgs'
```

(`VIRTUAL_ENV` must be unset so the build uses system `python3`, which has `empy`;
`pi/.venv` does not — see CLAUDE.md.)

`setup_franka_env.sh` also brings up the static IP via a **toggleable
NetworkManager profile** (`fr3-link`, created on first run with `autoconnect no`).
The wired port still does normal DHCP for other uses; the static IP is active only
while the profile is up. To revert manually: `nmcli connection down fr3-link`.
Override `FR3_IFACE` / `FR3_LAPTOP_IP` / `FR3_NM_PROFILE` before sourcing if the
hardware differs.

### NUC (`nu-crb`)

| | |
|---|---|
| OS | Ubuntu 22.04 Jammy |
| ROS2 | Humble |
| Laptop link | `enx00249b860356` = `10.0.0.2/24` |
| Robot link | `enp89s0` = `192.168.51.10/24`; FR3 at `192.168.51.20` |
| RMW | `rmw_cyclonedds_cpp` |
| `ROS_DOMAIN_ID` | unset → defaults to **0** |
| `CYCLONEDDS_URI` | `/home/franka/franka_ws/config/cyclonedds.xml` |

Bringup aliases (already configured on the NUC):

```bash
fr3-bringup        # ros2 launch franka_bringup franka.launch.py robot_ip:=192.168.51.20 arm_id:=fr3
fr3-arm-controller # ros2 run controller_manager spawner fr3_arm_controller \
                   #   -t joint_trajectory_controller/JointTrajectoryController \
                   #   --param-file .../franka_fr3_moveit_config/config/fr3_ros_controllers.yaml
```

### Shared DDS contract

Both machines must agree on all of:

- **RMW** `rmw_cyclonedds_cpp`.
- **`ROS_DOMAIN_ID` = 0** (the NUC leaves it unset, which is 0; the laptop sets it
  explicitly).
- **Unicast discovery only.** The NUC's `cyclonedds.xml` disables multicast and
  hardcodes the peer list `10.0.0.1` (laptop) and `10.0.0.2` (NUC). Therefore the
  **laptop must actually hold `10.0.0.1`** — there is no multicast fallback. If you
  use a different laptop IP, you must also edit the NUC's peer list.
- **Interface pinning.** Each side pins CycloneDDS to its NUC-link NIC
  (`enp0s31f6` on the laptop, `enx00249b860356` on the NUC) so discovery traffic
  doesn't leak onto WiFi or, later, the inference-server NIC.

`ros2_ws/config/cyclonedds_laptop.xml` is the laptop-side mirror of the NUC file.

### FR3 specifics

- **TF tree:** `base → fr3_link0 → … → fr3_link7 → fr3_link8 → fr3_hand → fr3_hand_tcp`.
  - Base frame: **`fr3_link0`**
  - EEF / tool frame: **`fr3_hand_tcp`** (tool center point, 0.1034 m past `fr3_hand`)
  - `policy_client_node` reads `base_frame` / `eef_frame` params (defaults above).
- **Gripper (Franka Hand):** see [Gripper interface](#gripper-interface-franka-hand) below — it
  behaves quite differently from the arm and has several traps.
- **Robot state:** `/franka_robot_state_broadcaster/current_pose` exposes the EEF
  pose as an alternative to the TF lookup, plus joint states / wrenches.
- **⚠ MoveIt planning group: use `fr3_arm`, NOT `fr3_manipulator`.** The SRDF defines
  both (`fr3_manipulator` tip = `fr3_hand_tcp`, `fr3_arm` tip = `fr3_link8`), and
  `fr3_manipulator` looks like the obvious choice since its tip is the TCP. But only
  `fr3_arm` has a kinematics solver entry in `franka_fr3_moveit_config/config/kinematics.yaml`,
  and Humble's `computeCartesianPath` needs one — with `fr3_manipulator` **every**
  Cartesian request returns `fraction=0.0` (error_code SUCCESS, 1 trajectory point).
  Verified on hardware: `fr3_arm` → `fraction=1.000`. `fr3_arm` still accepts
  `fr3_hand_tcp` as the target `link_name`, so we plan for the real TCP either way.
- **Humble `GetCartesianPath` has no velocity scaling.** `max_velocity_scaling_factor` /
  `max_acceleration_scaling_factor` don't exist on the request in Humble (added later);
  setting them raises `AttributeError`. `fr3_moveit_bridge` instead scales the planned
  trajectory in time (`_slow_trajectory`) before execution.

### Gripper interface (Franka Hand)

Launched by `fr3-bringup` automatically — `franka.launch.py`'s `load_gripper` defaults to `true`.

**It is action-only. There is no way to servo it.** `ros2 control list_hardware_interfaces` shows
**zero** finger/gripper interfaces: the hand is not in ros2_control at all, so the native
`cartesian_pose` command interface the arm exposes has no gripper counterpart. Nor is this a ROS
wrapper limitation — libfranka's `franka::Gripper` (`~/franka_ws/src/libfranka/include/franka/gripper.h`)
offers only `homing()`, `grasp()`, `move()`, `stop()`, `readOnce()`, all blocking and discrete.
This is why PolyUMI's gripper commander is deadbanded and rate-limited rather than a streaming
servo like UMI's — see [Phase 2.5](franka-inference-bringup.md#phase-25--gripper-control).

| Interface | Type | Notes |
|---|---|---|
| `/fr3_gripper/move` | `franka_msgs/action/Move` | `width` (m, **full aperture**), `speed` (m/s). Position only — applies no force, stalls on contact. |
| `/fr3_gripper/grasp` | `franka_msgs/action/Grasp` | `width`, `speed`, `force` (N), `epsilon.inner/outer`. The only action that actually **holds**: succeeds and keeps applying force if the final width lands in `[width-inner, width+outer]`. |
| `/fr3_gripper/gripper_action` | `control_msgs/action/GripperCommand` | Convenience wrapper: auto-dispatches `move()` when opening, `grasp()` when closing. **⚠ `position` is PER-FINGER** (the node does `width = 2 * position`), unlike Move/Grasp. Speed is fixed at `default_speed_` (0.1 m/s) and epsilon at 0.005 — both unsettable through this action. Out-of-range targets `abort()` rather than clamping. |
| `/fr3_gripper/homing` | `franka_msgs/action/Homing` | Empty goal; re-estimates max width. Needed after changing fingers. |
| `/fr3_gripper/stop` | `std_srvs/srv/Trigger` | The sanctioned way to interrupt; action `cancel` also calls `gripper_->stop()`. |

**No goal is ever rejected.** `gripper_action_server.cpp` returns `ACCEPT_AND_EXECUTE`
unconditionally and spawns a detached `std::thread` per goal — no queue, no preemption. libfranka
aborts the superseded command, surfacing as `goal_handle->abort()` with
`libfranka gripper: Command aborted!`. So "latest wins" holds, but every superseded goal reports a
failure. **Do not stream goals at the control rate.**

**State:** `/fr3_gripper/joint_states`, `name: [fr3_finger_joint1, fr3_finger_joint2]`. Each finger
reports **half** the aperture, so `width = position[0] + position[1]`. `velocity` and `effort` are
hardcoded `0.0` — there is no real force feedback. Measured rate **~17 Hz**, not the configured 30:
`publishGripperState()` calls the blocking `readOnce()` inside its timer callback, so the hand's UDP
stream is the real bound. Reachable from the laptop over DDS (verified). `max_width` (~0.0817 m
after homing) is **not published anywhere** — it exists only inside the node.

### Known upstream `franka_ros2` bugs (not fixed)

Both are real defects in `franka_ros2` v0.1.15, confirmed on this NUC. Neither is fixed here:
they live in the NUC's `~/franka_ws/src/franka_ros2` checkout — **not** in our
`external/franka_ros2` submodule, which is built for `franka_msgs` only, so patching this repo
would change nothing at runtime. A fix means editing NUC machine state plus
`colcon build --packages-select franka_gripper franka_bringup` and an `fr3-bringup` restart.
Recorded so nobody re-diagnoses them.

**1. The gripper's params file is silently ignored.**
`franka_gripper/config/franka_gripper_node.yaml` is keyed `franka_gripper:`, but
`gripper.launch.py` names the node `[arm_id, '_gripper']` = `fr3_gripper`, so the key never
matches. Verify with `ros2 param dump /fr3_gripper`: `state_publish_rate` reads **30** (the C++
default) rather than the YAML's 50, and `feedback_publish_rate` reads 10 rather than 30. Only
`robot_ip` / `joint_names` take effect, because those are passed as an inline dict.
Impact on us is small — `Move`/`Grasp` take speed and epsilon per goal, so our bridge sets what it
needs. And fixing it would **not** raise the observed ~17 Hz state rate, which is bounded by the
blocking `readOnce()`, not by the timer.

**2. There is no finger TF.** `franka.launch.py:147` points `joint_state_publisher` at
`franka_gripper/joint_states`, while the node actually publishes `/fr3_gripper/joint_states`
(`ros2 topic info -v /franka_gripper/joint_states` → `Publisher count: 0`). Finger joints therefore
never reach `robot_state_publisher`, so `fr3_leftfinger` / `fr3_rightfinger` do not resolve in TF.
Root cause is one level up: `franka.launch.py` forwards only `robot_ip` and `use_fake_hardware` to
`gripper.launch.py`, never `arm_id` — the gripper's own default happens to be `fr3`, so our topic
names line up by coincidence and would break for any other `arm_id`.
Harmless for PolyUMI because `policy_client_node` subscribes to `/fr3_gripper/joint_states`
directly, but it will bite anyone expecting finger frames in TF.

### Quick checks

```bash
# laptop, after `source setup_franka_env.sh` and with `fr3-bringup` up on the NUC:
ping 10.0.0.2
ros2 node list                                   # NUC nodes appear
ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp # live transform
```

### rmw version mismatch — what is and isn't harmless

The two sides run different `rmw_cyclonedds_cpp` majors — **NUC 1.3.4** (Humble) vs
**laptop 4.0.2** (Kilted) — though the CycloneDDS core is the same (0.10.5). rmw 4.x
encodes a **type hash** into DDS discovery `USER_DATA`; rmw 1.3.x predates that and
can't parse it. This surfaces as two loud messages:

- On the **laptop**, once per discovered remote topic:
  `[WARN] [rmw_cyclonedds_cpp]: Failed to parse type hash for topic '...' from USER_DATA '(null)'.`
- On the **NUC**, when a laptop `ros2` node appears:
  repeated `'invalid data size'` / `'string data is not null-terminated', at .../serdata.cpp`.

**Harmless for small/simple messages.** Verified: `ros2 topic hz /joint_states` gives a
real rate on the laptop, TF crosses fine, and a `geometry_msgs/PoseStamped` published
laptop→NUC arrives byte-for-byte intact. `trajectory_msgs/JointTrajectory` (the gripper chunk on
`/polyumi/target_gripper`) was checked the same way and also **crosses intact** — the NUC received
`frame_id`, `joint_names`, and every point's `positions` + `time_from_start` exactly as published,
with the `serdata.cpp:384` noise appearing alongside but not corrupting the payload. The inference
loop's observation path is unaffected. rmw_cyclonedds 4.0.2 has no switch to suppress the type-hash
emission (it only reads `CYCLONEDDS_URI`), so we accept this noise.

**⚠ NOT harmless for large nested messages.** A `MoveGroup.Goal` sent from the **laptop**
to the NUC's move_group fails: move_group logs `Catastrophic failure` right next to those
same `serdata.cpp:384` errors, and the client gets `error_code=99999` (not a real
MoveItErrorCode). The goal is arriving corrupted. Small service calls (`/compute_fk`,
`/check_state_validity`, `/get_planning_scene`) cross fine and return real data, so the
boundary is roughly message size/nesting — which makes this confusing to diagnose.

**This is why `fr3_moveit_bridge` runs on the NUC** rather than the laptop calling
move_group directly: keeping the MoveIt calls same-rmw (NUC-local) sidesteps the question
entirely, and is known-good (that's the configuration that actually moves the arm).

Scope of what was actually tested, so nobody over-reads this: the `MoveGroup.Goal`
corruption is confirmed. Whether a laptop-side `GetCartesianPath` call would survive is
**untested** — we hit the separate `fr3_manipulator` planning-group bug (below) first,
which produces its own `fraction=0.0` *both* laptop-side and NUC-local, then moved the
MoveIt calls to the NUC and never retried Cartesian from the laptop. Don't assume the
laptop path is fine for Cartesian just because the group bug explains that symptom.

If `ros2 topic hz` ever hangs, it's almost certainly **not** this — check whether the
publisher is actually running (e.g. the Pi stream for `/pi/*`).

## Running Demos & Inference

TODO describe setup & connection of devices.

This brings up the inference loop: the FR3 stack on the NUC, the PolyUMI nodes +
`policy_client_node` on the laptop, and an inference server. The server is either the **real**
trained policy (`serve_policy.sh` on the GPU box — step 2) or the **dummy** oscillator
(`dummy-server` on the laptop — step 2, alternative). The client pulls the live EEF pose from the
NUC's TF over DDS, POSTs observations to the server, logs the returned 8-vector action chunk, and
publishes it to `/polyumi/target_poses_preview` for Foxglove (always) and — only with
`execute_motion:=true` — to `/polyumi/target_poses` for the NUC bridge to execute.

Start the pieces in separate terminals, in this order.

> **Shortcut: `./fr3_session.sh`** builds this entire wall as one tmux session — NUC, Pi, GPU
> box, and laptop — with the safe commands already running and the robot-moving ones typed at
> the prompt for you to confirm. The steps below are what it automates, and remain the
> reference for doing it by hand or debugging a pane that misbehaves. See
> [Session launcher](#session-launcher-fr3_sessionsh).

### **1. NUC — bring up the FR3** (enable FCI on the Desk UI first):

```bash
ros2 launch nuc/launch/fr3_bringup.launch.py   # franka_bringup + fr3_arm_controller spawner
```

This is the **hardware session**: `franka_bringup` plus the joint-trajectory controller
move_group executes through. It replaces the old two-terminal `fr3-bringup` +
`fr3-arm-controller` pair — those were only ever split because the controller spawner has to
run *after* `controller_manager` exists, not because they are independent. (The spawner exits
once the controller is active; it never needed a terminal of its own.) The aliases still work
if you want the pieces separately.

Kept deliberately separate from step 1b so it can be **restarted on its own** — this is the
component that crashes mid-session (see [TF lookup fails](#tf-lookup-fails-fr3_link0--does-not-exist--no-tf-at-all--fr3-bringup-crashed)),
and the one gated on enabling FCI by hand.

Step **1b** is **only needed to actually move the arm** (the Phase 2
`execute_motion:=true` path). The log-only inference loop skips it.

Both launch files run on the NUC from a clone of this repo, in their own terminals. Each needs
the NUC's ROS + DDS env — a non-interactive shell does **not** source `~/.bashrc`, and
without `CYCLONEDDS_URI` the node comes up on the wrong RMW and is invisible to
everything else (this is one reason `fr3_session.sh` opens a *tmux* on the NUC rather than
running commands over a bare `ssh`):

```bash
source /opt/ros/humble/setup.bash
source ~/franka_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/franka_ws/config/cyclonedds.xml
```

#### **1b. NUC — start the inference stack** (second NUC terminal):

```bash
ros2 launch nuc/launch/fr3_inference.launch.py                        # dry run, nothing moves
ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true  # fingers only
ros2 launch nuc/launch/fr3_inference.launch.py \
    execute_arm:=true execute_gripper:=true max_velocity_scaling:=0.2
```

Starts **move_group + both PolyUMI bridges** — the three things that sit on top of the
hardware session. They start, fail, and restart together without touching the arm's state,
which is why they share one launch file where step 1 gets its own.

**Two execute flags, not one**, and both default false: launching this file never moves the
robot on its own. Keeping them separate is what makes the first-run-on-hardware sequence below
possible (gripper executing, arm planning-only). The three components are described next.

**move_group** adds **only** the planner (exposing `/move_action`,
`/execute_trajectory`, `/compute_cartesian_path`) — no controllers or
robot_state_publisher — so it runs alongside the already-up `fr3-bringup` without
collision. Expect a harmless `No 3D sensor plugin(s) defined for octomap updates` error
(we have no depth camera, so no environment collision geometry).

We ship our own copy in [`nuc/`](../nuc/) rather than using
`franka_fr3_moveit_config`'s, because upstream's `move_group.launch.py` is unusable:
it references `robot_ip` / `use_fake_hardware` / `fake_sensor_commands` without
declaring them (`launch configuration 'fake_sensor_commands' does not exist`), and it
omits the params move_group needs to be functional — the OMPL pipeline, the controller
list, and the planning-scene monitor. Ours declares the args and passes those params
(copied from that package's own `moveit.launch.py`). Without them move_group defaults to
CHOMP, logs `No controller_names specified`, and cannot execute. Sanity-check the log for:

```
Using planning interface 'OMPL'
Added FollowJointTrajectory controller for fr3_arm_controller
Trajectory execution is managing controllers
```

(Do **not** use upstream `moveit.launch.py` — it starts a *second* controller_manager +
robot_state_publisher and collides with `fr3-bringup`.)

**`fr3_moveit_bridge`** (`execute_arm`) subscribes `/polyumi/target_poses` (a `PoseArray` — one
action chunk) and drives the local
move_group, planning the whole chunk as a single multi-waypoint Cartesian path.
`max_velocity_scaling` (default `0.1`, max `1.0` = the full speed
move_group already planned at) time-scales the trajectory; **start low** (e.g. `0.1`–`0.3`)
with a hand on the e-stop, then raise it once you trust the motion. It logs
`move_group found (compute_cartesian_path ready).` at
startup — if it instead says `NOT found after 10s`, move_group failed to come up (check the
same launch file's output).

**`fr3_gripper_bridge`** (`execute_gripper`) subscribes `/polyumi/target_gripper` (a
`trajectory_msgs/JointTrajectory` — the width half of the
action chunk) and drives `/fr3_gripper/{move,grasp}`. With its flag false it logs the goal it
would send and commands nothing. Independent of `move_group`, so it works even if move_group
failed to start.

Because the hand cannot be servoed (see [Gripper interface](#gripper-interface-franka-hand)), this
node deliberately does **not** track every chunk: it deadbands (`width_deadband_m`, default 5 mm)
and rate-limits (`min_command_period_s`, default 0.25 s), sending only the latest desired width.
A quiet log with occasional goals is correct; a stream of `Command aborted!` is not.

The deadband is measured against the width the hand actually **accepted**, not the last one
attempted — so a goal that never lands (action server gone, goal rejected) is retried on the next
tick rather than being deadbanded away, which would otherwise park the fingers at a width they
never received. Parameters are validated at startup and the node refuses to start on a bad one;
`min_command_period_s: 0` in particular used to divide by zero inside the timer callback.

**First time on hardware, launch with `execute_gripper:=true execute_arm:=false`**, so a bad
width command moves fingers and nothing else. Keep the hand clear of objects and of the table.
`fr3_session.sh` pre-types this line for you but with `execute_arm:=true` — it is pre-typed,
not run, precisely so you can edit the flags before pressing Enter.

### **2. Inference server — real (GPU box) or dummy (laptop).**

*Real policy* — on the GPU workstation (`sheep`), serve a trained checkpoint. `serve_policy.sh`
builds the image and wires the rootless-Docker flags + checkpoint/HF-cache mounts (see
[training-instructions.md](training-instructions.md)):

```bash
# on the GPU box:
CKPT=/abs/path/to/epoch=0070-train_loss=0.021.ckpt ./serve_policy.sh
# from the laptop, confirm it's reachable:
curl http://<gpu-host>:8000/health           # -> {"status":"ready", ...}
```

The laptop reaches it over LAN; pass its URL to the launch in step 4
(`inference_server_url:=http://<gpu-host>:8000/predict_cartesian/`). Do **not** run
`external/polyumi_diffusion_policy/docker/serve.sh` on the host — it is the in-container
entrypoint and fails with `exec: uvicorn: not found`.

*Dummy oscillator* — alternative for wiring/CI with no GPU or checkpoint (its own laptop terminal):

```bash
cd inference_server
uv run dummy-server   # FastAPI on 0.0.0.0:8000; oscillates X around HOME_POSE
```

`inference_server` is its own isolated uv project (not part of the repo
workspace), so `uv run` here creates/uses a standalone `inference_server/.venv`
with only fastapi/uvicorn/numpy — no need to source anything. The command is
`dummy-server` (hyphen), the `[project.scripts]` entry point. Leave
`inference_server_url` at its default (`http://localhost:8000/predict_cartesian/`) in step 4.

### **3. Pi — start the camera/audio stream** (ssh into the Pi):

```bash
polyumi-pi stream   # ZMQ PUSH: video on :5555, audio on :5556
```

`pi_receiver_node` (started by the launch in step 4) pulls these over ZMQ and
republishes them as `/pi/*`. Without this running, Foxglove shows no Pi feed, and `pi_receiver_node`
logs a warning. (The FR3 inference loop itself doesn't depend on the Pi, but the full
demo does.)

### **4. Laptop — PolyUMI ROS2 nodes + policy client** (another terminal):

```bash
source setup_franka_env.sh          # CycloneDDS + domain 0 + bring up the fr3-link NM profile
cd ros2_ws
source install/setup.bash           # (build first if needed: colcon build)
ros2 launch polyumi_ros2 inference_demo.launch.xml pi_host:=<raspberry pi IP address>
# default inference_server_url is http://localhost:8000/predict_cartesian/
# To MOVE the arm (Phase 2), add: execute_motion:=true
#   -> publishes each action chunk on /polyumi/target_poses; needs step 1b on the NUC with
#      execute_arm:=true. Speed is set there (max_velocity_scaling), not here.
#   Chunk size is n_action_steps (default 8) -- see "Action-chunk execution" above.
# Default is log-only: actions are logged, no pose published, arm does not move.
# To iterate on FR3 motion alone without the Pi running, add: motion_only:=true
#   -> skips pi_receiver_node (no ZMQ connection attempt to the Pi). GoPro + foxglove
#   still run (policy_client_node needs the GoPro image to fill its observation buffer).
```

**Real-policy dry run (recommended first pass on the arm — no motion, just watch the commanded
chunk in Foxglove):**

```bash
ros2 launch polyumi_ros2 inference_demo.launch.xml motion_only:=true \
    inference_server_url:=http://<gpu-host>:8000/predict_cartesian/ \
    max_image_age_s:=0.3 \      # tolerate the Elgato's ~200 ms 1080p convert latency
    tf_use_latest:=true         # ONLY if the laptop<->NUC clocks are skewed; static arm only
```

`execute_motion` stays false, so nothing moves; the commanded chunk is published to
`/polyumi/target_poses_preview` (add it in Foxglove — pose arrows in `fr3_link0`). Sane output sits
near the current EEF with small step-to-step deltas. `tf_use_latest` and `max_image_age_s` are
workarounds — see [Troubleshooting](#troubleshooting); drop them once the clock is synced and a
faster camera path is in place, which are prerequisites for `execute_motion:=true`.

**Testing motion without the full loop.** To move the arm through one chunk by hand
(rather than the 10 Hz dummy oscillation), skip step 4 and publish a `PoseArray` directly
from the laptop — read the current pose, then target a small offset. A single-pose array
is a valid (trivial) chunk:

```bash
ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp     # note x,y,z + quat, then Ctrl-C
ros2 topic pub -1 /polyumi/target_poses geometry_msgs/msg/PoseArray \
  "{header: {frame_id: fr3_link0}, poses: [{position: {x: 0.322, y: -0.001, z: 0.446}, \
    orientation: {x: -1, y: 0, z: 0, w: 0}}]}"
```

Use your measured pose with ~2 cm added to one axis (`-1` publishes once). The bridge
should log `Executed chunk (1 waypoints).` and the arm should creep to it.

### Session launcher (`fr3_session.sh`)

Steps 1–4 as one tmux session, from the repo root:

```bash
./fr3_session.sh                # create, or re-attach if it is already up
SKIP_DEPLOY=1 ./fr3_session.sh  # ...without re-syncing the NUC/Pi source trees first
./fr3_session.sh --kill-local   # tear down the LOCAL session only (remote ones survive)
./fr3_session.sh --kill         # --kill-local, plus stop the remote sessions too
```

Three windows: `nuc` (bringup | inference stack), `polyumi-pi` (Pi | GPU box), `laptop`.

**Every fresh start (not a re-attach) deploys first.** `nuc/` is rsynced to the NUC, and
`./deploy.sh` (see CLAUDE.md / README.md, also runnable standalone: `./deploy.sh <pi_ssh_host>`)
is called for the Pi — so what runs on both machines matches this working copy, not whatever
they last had checked out. Sheep is deliberately excluded: it tracks its own training branch,
and force-syncing it would silently swap out the checkpoint code from under you.
Skip both with `SKIP_DEPLOY=1` once you know they're already current — useful for a fast
re-launch while iterating on the tmux layout itself rather than on NUC/Pi code. Each target is
independent and non-fatal: an unreachable Pi (powered off, say) warns and is skipped rather
than blocking the NUC and laptop panes from coming up.

**Safe commands run; robot-moving ones are typed at the prompt and left for you to press
Enter on.** So bringup and the Pi stream start themselves, while the inference stack (carries
the execute flags), the policy server (carries the checkpoint path), and the laptop client
(depends on everything above, and there is no readiness gate) wait for you. Nothing in the
script can move the robot on its own.

**The NUC and GPU-box panes run tmux on the remote host**, not a bare ssh — so a laptop sleep
or wifi blip costs nothing (re-run the script to re-attach, everything is still running), and
the interactive shell means `CYCLONEDDS_URI` and the `fr3-*` aliases are actually set, which a
bare `ssh host 'cmd'` would silently skip. The Pi is a plain ssh: stateless, cheap to restart.
Consequences worth knowing:

- `--kill-local` only kills the **local** session — the NUC and GPU-box sessions survive by
  design, for the re-attach case. `--kill` also stops those specific remote sessions
  (`tmux kill-session`, not `kill-server`, so any unrelated session on that host is left alone).
- Those panes are **nested tmux**, so `C-b` goes to the outer one. `C-b C-b` sends a prefix
  through to the inner session.
- **Re-attaching never types into a live remote pane.** The script probes each remote session
  first and leaves the ones already running completely alone — a shell mid-bringup, or holding
  a pre-typed line you have not pressed Enter on, is handed back untouched. (`send-keys`
  *appends* to a readline buffer rather than replacing it, so a second pass would otherwise
  concatenate two commands and submit the result.)
- A host with no tmux installed, or one that is not answering, degrades to a plain `ssh` with
  a warning rather than failing — one machine being down should not block the others. That
  pane just will not survive a disconnect.

The Pi's address is resolved from your ssh config (`ssh -G $PI_SSH_HOST`) at launch rather than
hardcoded, since it is on DHCP and does move. `PI_SSH_HOST` defaults to `polyumi-pi` — the alias
other users are expected to set up — so if yours is named differently, override it:
`PI_SSH_HOST=conorpi ./fr3_session.sh`. Repo paths and URLs are further environment overrides at
the top of the script: `NUC_REPO`, `SHEEP_REPO`, `INFERENCE_URL`, `MAX_IMAGE_AGE_S`,
`SHELL_SETTLE_S` (raise the last one if pre-typed lines land mangled — typing races the
remote shell's startup).

## Troubleshooting

### Nothing publishes / Foxglove shows nothing — a duplicate or leftover launch (most common)
Symptoms: `foxglove_bridge` aborts at startup with
`terminate called ... Couldn't initialize websocket server: Bind Error`, and/or the GoPro frames
stall a few seconds in — `Dropped control tick: newest camera frame is N ms old` with **N climbing
without bound** — so `policy_client_node` stops posting and the preview topic goes quiet. Cause:
another (or leftover) launch already holds port **8765** (foxglove) and **`/dev/video2`** (the
camera); two processes can't share either, so the second foxglove aborts and the two camera nodes
starve each other. Clear leftovers and confirm a single stack before relaunching:

```bash
pkill -f "ros2 launch"; pkill -f foxglove_bridge; pkill -f v4l2_camera
ros2 node list      # expect only NUC nodes (or nothing) on the laptop before you launch
```

### TF lookup fails: "extrapolation into the past" — laptop↔NUC clock skew
`policy_client_node` logs `TF lookup failed: Lookup would require extrapolation into the past`, with
the "earliest data" time far *ahead* of the requested time. The NUC's wall clock is ahead of the
laptop's, so NUC-stamped TF lands outside the buffer. (This is the first path that reads NUC TF *on
the laptop* — Phase 2 execution runs entirely in the NUC's own time domain, so it never exposed the
skew.) Two fixes:
- **Proper (required before execution):** sync the clocks. The NUC is on the isolated `10.0.0.x`
  link, usually with no internet NTP — point its chrony at the laptop (`server 10.0.0.1 iburst`,
  with `allow 10.0.0.0/24` on the laptop's chrony) or set it manually.
- **Dry run only:** `tf_use_latest:=true` looks up the latest EEF transform (tf2 time=0) instead of
  the latency-aligned instant. Valid only while the arm is **stationary** (`execute_motion:=false`);
  do NOT use it for execution — a moving arm needs the time-aligned pose.

### TF lookup fails: "fr3_link0 ... does not exist" / no TF at all — fr3-bringup crashed
If `ros2 topic info /tf_static` shows `Publisher count: 0` and `tf2_echo fr3_link0 fr3_hand_tcp`
reports the frame doesn't exist, the NUC's `fr3-bringup` has died (it can crash mid-session).
Restart `fr3-bringup` (and `fr3-arm-controller`) on the NUC; TF returns within a second or two.

### Every tick dropped: "capture pipeline stalled" — camera latency, not a stall
The Elgato HD60 X presents the GoPro feed as **1080p YUYV**, and `v4l2_camera` does a *software*
YUYV→RGB conversion (it logs "possibly slow conversion") into a ~6 MB `rgb8` message — adding
~200 ms of stamp-to-usable latency, past the default ~50 ms freshness limit. Raise it with
`max_image_age_s:=0.3`. This only tolerates older frames; image and pose stay aligned to the
frame's capture stamp, so it's safe for the dry run. For **execution**, prefer a genuinely faster
camera path (lower published resolution, or the compressed transport) so the policy isn't acting on
200 ms-old vision.

### Gripper: a stream of `Command aborted!` / "Gripper move failed"
`franka_gripper` accepts every goal and never preempts; libfranka aborts whichever command a new
one supersedes, so each superseded goal ends ABORTED. A steady stream of these means
`fr3_gripper_bridge` is sending goals far too fast — check that `min_command_period_s` and
`width_deadband_m` are actually applied (a deadband of 0 with a noisy commanded width will fire
every period). Occasional aborts when the width changes quickly are expected and logged at info.

### Gripper never moves
In order: is `fr3_gripper_bridge` running with `execute:=true` (it defaults to false)? Does
`ros2 action list | grep fr3_gripper` show the four servers — if not, `fr3-bringup` was started with
`load_gripper:=false`. Is anything arriving on `/polyumi/target_gripper` (`ros2 topic hz`)? If the
laptop publishes but the NUC sees nothing, suspect the `JointTrajectory` message crossing the
rmw-version gap — compare against `/polyumi/target_poses`, which is known to cross fine. Finally,
a commanded width inside `width_deadband_m` of the current one is *intentionally* not sent.
(The `JointTrajectory` message itself is known to cross the laptop↔NUC rmw gap intact — that has
been verified, so it is not the likely culprit.)

### First inference times out, then recovers; many actions dropped as stale
The first `/predict_cartesian/` after the server starts includes GPU/model warmup and can exceed
the POST timeout (`POST ... failed: timed out`). It self-recovers next tick; raise `post_timeout_s`
if it persists. Real diffusion inference is ~200–500 ms/call, so the stale-drop logic discards most
of each 8-step chunk (`dropped 5/8 …`, occasionally `dropped all 8`). Fine for a dry run; before
execution, cut latency (fewer diffusion steps, compressed image transport, or a larger
`n_action_steps`).

Confirm the loop is live: `policy_client_node` logs one `episode /reset sent` line, then
`action chunk n=… (dropped …/… stale, inference=…ms) first: x=… y=… z=… grip=…` each tick, and
Foxglove (`ws://localhost:8765`, using the config in `ros2_ws/src/polyumi_ros2/foxglove/layouts/stream_demo.json`)
shows the GoPro, the Pi camera/audio, FR3 TF, and the commanded chunk on
`/polyumi/target_poses_preview`. If the client warns about TF lookups, re-check the
[Quick checks](#quick-checks) above — the NUC must be reachable and `fr3-bringup`
running — and see [Troubleshooting](#troubleshooting).
