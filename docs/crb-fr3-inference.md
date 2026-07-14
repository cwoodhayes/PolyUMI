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
  - foxglove_bridge                               - fr3-bringup (franka_bringup, arm_id:=fr3)
  - v4l2_camera (GoPro)                           - fr3-arm-controller
  - pi_receiver_node                              - move_group  (nuc/launch/fr3_move_group.launch.py)
  - policy_client_node ──HTTP──┐                  - fr3_moveit_bridge  (nuc/fr3_moveit_bridge.py)
  - dummy_server (localhost:8000) ◄┘              - publishes fr3_* TF + joint states
        │                                         - enp89s0 = 192.168.51.10 → robot @ .20
        └── /polyumi/target_poses (PoseArray) ──────────► fr3_moveit_bridge ──► move_group
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
- **Gripper:** width on `/fr3_gripper/joint_states`; action servers
  `/fr3_gripper/{grasp,move,gripper_action,homing}`. (Wired into observations /
  execution in Phase 2; currently a `0.0` placeholder.)
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
laptop→NUC arrives byte-for-byte intact. The inference loop's observation path is
unaffected. rmw_cyclonedds 4.0.2 has no switch to suppress the type-hash emission (it
only reads `CYCLONEDDS_URI`), so we accept this noise.

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

This brings up the **dummy** inference loop (no real checkpoint): the FR3 stack on
the NUC, the PolyUMI nodes + `policy_client_node` on the laptop, and the dummy
server (currently also on the laptop). At the end the client logs 8-vector actions
at 10 Hz, pulling the live EEF pose from the NUC's TF over DDS.

Start the pieces in separate terminals, in this order.

**1. NUC — bring up the FR3** (enable FCI on the Desk UI first):

```bash
fr3-bringup          # franka_bringup, arm_id:=fr3, robot @ 192.168.51.20
fr3-arm-controller   # in a second terminal: spawn the joint-trajectory controller
```

Steps **1b** and **1c** are **only needed to actually move the arm** (the Phase 2
`execute_motion:=true` path). The log-only inference loop skips them.

Both run on the NUC from a clone of this repo, in their own terminals. Each needs the
NUC's ROS + DDS env — a non-interactive shell does **not** source `~/.bashrc`, and
without `CYCLONEDDS_URI` the node comes up on the wrong RMW and is invisible to
everything else:

```bash
source /opt/ros/humble/setup.bash
source ~/franka_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/franka_ws/config/cyclonedds.xml
```

**1b. NUC — start MoveIt `move_group`** (third NUC terminal):

```bash
ros2 launch <repo>/nuc/launch/fr3_move_group.launch.py robot_ip:=192.168.51.20
```

This adds **only** the `move_group` planner (exposing `/move_action`,
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

**1c. NUC — start the MoveIt bridge** (fourth NUC terminal):

```bash
python3 <repo>/nuc/fr3_moveit_bridge.py --ros-args -p execute:=true -p max_velocity_scaling:=0.05
```

Subscribes `/polyumi/target_poses` (a `PoseArray` — one action chunk) and drives the local
move_group, planning the whole chunk as a single multi-waypoint Cartesian path.
**`execute` defaults to `false`** (plan-only, no motion) — pass `execute:=true` to
actually move the arm. `max_velocity_scaling` (default `0.1`, max `1.0` = the full speed
move_group already planned at) time-scales the trajectory; **start low** (e.g. `0.1`–`0.3`)
with a hand on the e-stop, then raise it once you trust the motion. It logs
`move_group found (compute_cartesian_path ready).` at
startup — if it instead says `NOT found after 10s`, step 1b isn't running.

**2. Laptop — dummy inference server** (its own terminal):

```bash
cd inference_server
uv run dummy-server   # FastAPI on 0.0.0.0:8000; oscillates X around HOME_POSE
```

`inference_server` is its own isolated uv project (not part of the repo
workspace), so `uv run` here creates/uses a standalone `inference_server/.venv`
with only fastapi/uvicorn/numpy — no need to source anything. The command is
`dummy-server` (hyphen), the `[project.scripts]` entry point.

**3. Pi — start the camera/audio stream** (ssh into the Pi):

```bash
polyumi-pi stream   # ZMQ PUSH: video on :5555, audio on :5556
```

`pi_receiver_node` (started by the launch in step 4) pulls these over ZMQ and
republishes them as `/pi/*`. Without this running, Foxglove shows no Pi feed, and `pi_receiver_node`
logs a warning. (The FR3 inference loop itself doesn't depend on the Pi, but the full
demo does.)

**4. Laptop — PolyUMI ROS2 nodes + policy client** (another terminal):

```bash
source setup_franka_env.sh          # CycloneDDS + domain 0 + bring up the fr3-link NM profile
cd ros2_ws
source install/setup.bash           # (build first if needed: colcon build)
ros2 launch polyumi_ros2 inference_demo.launch.xml pi_host:=<raspberry pi IP address>
# default inference_server_url is http://localhost:8000/predict_cartesian/
# To MOVE the arm (Phase 2), add: execute_motion:=true
#   -> publishes each action chunk on /polyumi/target_poses; needs steps 1b + 1c on the NUC.
#   Speed is set on the BRIDGE (step 1c max_velocity_scaling), not here.
#   Chunk size is n_action_steps (default 8) -- see "Action-chunk execution" above.
# Default is log-only: actions are logged, no pose published, arm does not move.
# To iterate on FR3 motion alone without the Pi running, add: motion_only:=true
#   -> skips pi_receiver_node (no ZMQ connection attempt to the Pi). GoPro + foxglove
#   still run (policy_client_node needs the GoPro image to fill its observation buffer).
```

**Testing motion without the full loop.** To move the arm through one chunk by hand
(rather than the 10 Hz dummy oscillation), skip step 4 and publish a `PoseArray` directly
from the laptop — read the current pose, then target a small offset. A single-pose array
is a valid (trivial) chunk:

```bash
ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp     # note x,y,z + quat, then Ctrl-C
ros2 topic pub -1 /polyumi/target_poses geometry_msgs/msg/PoseArray \
  "{header: {frame_id: fr3_link0}, poses: [{position: {x: 0.322, y: -0.001, z: 0.446}, \
    orientation: {x: 0.999, y: -0.010, z: -0.054, w: 0.002}}]}"
```

Use your measured pose with ~2 cm added to one axis (`-1` publishes once). The bridge
should log `Executed chunk (1 waypoints).` and the arm should creep to it.

Confirm the loop is live: `policy_client_node` logs `action x=… y=… z=… grip=…`
at ~10 Hz, and Foxglove (`ws://localhost:8765`, using the config in `ros2_ws/src/polyumi_ros2/foxglove/layouts/stream_demo.json`) shows the GoPro, the Pi
camera/audio, and FR3 TF. If the client warns about TF lookups, re-check the
[Quick checks](#quick-checks) above — the NUC must be reachable and `fr3-bringup`
running.
