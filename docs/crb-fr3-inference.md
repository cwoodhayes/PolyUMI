# CRB FR3 Inference Setup

**Note for Northwestern CRB members**: This document describes how to run inference on the CRB lab's Franka FR3 arm (in the student office, connected to the NUC with the skull on it). 

**Note for users outside of Northwestern**: This is specific to our equipment, but is likely still useful as an example to bring up an inference setup on your own arm.

**Everything from here down is for an assumed audience of "people in Northwestern CRB with access to this setup.**

In general, to start exploring yourself, bring up the session as described below, after which the following commands are your friends:
`ros2 topic list`, `ros2 action list`, `ros2 control list_controllers`, `ros2 param dump <node>`. See also the launch files in [`nuc/launch/`](../nuc/launch/) and
[`ros2_ws/src/polyumi_ros2/launch/`](../ros2_ws/src/polyumi_ros2/launch/). 

One note - my SSH alias for the NUC is "jailfranka", which is referred to sometimes in the docs below. 

## Bringing up inference session

### Convenience session bringup script

The following script at the root of the repo enables bringing up all of the functionality described below in a convenient tmux session. I'd recommend just starting there. 

```bash
# step 1: power on the arm, unlock it, and enable FCI
# step 2: run the following:
./fr3_session.sh
```

Safe commands run themselves; robot-moving ones are pre-typed for you to press Enter on. It
rsyncs `nuc/` to the NUC and runs `./deploy.sh` for the Pi on every fresh start. 

The script is essentially performing the following steps for you:

## 1. NUC — hardware

Enable FCI in the Desk UI first. Then, in an **interactive** shell (a bare `ssh host 'cmd'` skips
`~/.bashrc`, so `CYCLONEDDS_URI` goes unset and the node comes up invisible to everything else):

```bash
ros2 launch nuc/launch/fr3_bringup.launch.py     # franka_bringup + fr3_arm_controller spawner
```

Kept separate from step 2 because this is the piece that crashes mid-session (the arm's own
safety stops kill `ros2_control_node`, which is `required`) and the one gated on FCI, so it has
to be restartable alone.

## 2. NUC — inference stack

```bash
ros2 launch nuc/launch/fr3_inference.launch.py                        # nothing moves
ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true  # fingers only
ros2 launch nuc/launch/fr3_inference.launch.py \
    execute_arm:=true execute_gripper:=true max_velocity_scaling:=0.2
```

**Both execute flags default false** — launching alone never moves the robot. 

Velocity scaling is only applicable if the arm is being controlled by moveit, which is deprecated aside from the homing functionality.

## 3. Inference server

```bash
# real policy, on the GPU box (see training-instructions.md):
CKPT=/abs/path/to/<name>.ckpt ./serve_policy.sh
curl http://<gpu-host>:8000/health            # from the laptop

# or the dummy oscillator — no GPU, no checkpoint:
cd inference_server && uv run dummy-server    # :8000
```

`external/polyumi_diffusion_policy/docker/serve.sh` is the *in-container* entrypoint; you should not run this directly.

## 4. Pi

SSH into the raspberry pi and run:

```bash
polyumi-pi stream
```

## 5. Laptop

```bash
source setup_franka_env.sh          # RMW, domain 0, CYCLONEDDS_URI, the fr3-link static IP
cd ros2_ws && source install/setup.bash
ros2 launch polyumi_ros2 inference_demo.launch.xml pi_host:=<pi IP>
```

Default is **log-only** — actions are logged and previewed on `/polyumi/target_poses_preview`,
the arm does not move. `execute_motion:=true` publishes the chunk for the NUC to execute.
`ros2 launch ... --show-args` lists the rest; `motion_only:=true` (skip waiting for/using the Pi) is also useful.

Recommended first pass on the arm: real server, `execute_motion` false, watch the preview poses
in Foxglove (`ws://localhost:8765`, layout in `ros2_ws/src/polyumi_ros2/foxglove/layouts/`).
Sane output sits near the current EEF with small step-to-step deltas.

Health and latency scalars are published one-per-topic under `/polyumi/diag/*` for Foxglove's
Plot panel — `ros2 topic list | grep diag` for the set. **`n_published_arm` is the one to watch**:
the higher the number the better (this is the number of actions we didn't have to discard due to latency in each chunk).

## Architecture on the NUC

`fr3_bringup` owns the hardware: `franka_bringup` (controller_manager + the libfranka hardware
interface), the `fr3_arm_controller` spawner, `robot_state_publisher`, and a
`static_transform_publisher` for `polyumi_tcp`.

`fr3_inference` adds the three things that sit on top of it — move_group, `fr3_moveit_bridge`,
`fr3_gripper_bridge` — plus the `polyumi_cartesian_impedance_controller` spawner, spawned
`--inactive`. They start, fail and restart together without touching the arm's state.

The laptop publishes an entire inference **action chunk** (`n_action_steps` waypoints, not
`actions[0]`) and a NUC-side executor drives the arm from it. Two executors exist; the laptop's
`wire` param picks the message format and the NUC's `executor` arg picks the consumer, and the
two must agree:

| `wire` / `executor` | topic | consumer |
|---|---|---|
| `multidof` (default) | `/polyumi/target_poses_traj` | `polyumi_cartesian_impedance_controller` — 1 kHz streaming servo |
| `pose_array` | `/polyumi/target_poses` | `fr3_moveit_bridge` → move_group, one Cartesian plan per chunk |
| always | `/polyumi/target_gripper` | `fr3_gripper_bridge` → `/fr3_gripper/{move,grasp}` |

A mismatch is loud rather than silent: nothing subscribes, the arm holds still, and the client
warns every second naming the topic it expected. The full contract is the module docstring of
`ros2_ws/src/polyumi_ros2/polyumi_ros2/target_chunk.py`.

Both controllers claim the same `<joint>/effort` interfaces, so **exactly one holds the arm** —
`ros2 control list_controllers` tells you which:

The moveit (pose_array) executor is deprecated and no longer intended for use.

```bash
# Arm must be STATIONARY: switching restarts the libfranka control loop.
ros2 control switch_controllers --deactivate fr3_arm_controller \
    --activate polyumi_cartesian_impedance_controller
```

`/polyumi/home` does this swap itself, both directions, and hands the arm back afterwards.

## The facts you can't deduce by looking

- **`polyumi_tcp`, not `fr3_hand_tcp`**, is the policy's frame: the closed-fingertip midpoint in
  GoPro-optical axes. Defined once in [`nuc/tcp_calib.py`](../nuc/tcp_calib.py) and reaching TF
  and move_group's RobotModel from there. The stock `fr3_hand_tcp` is a different physical point.
  Verify on hardware with `ros2 run polyumi_ros2 tcp_pivot_test` (pivots about the TCP with the
  gripper closed — the fingertips should hold still). **Moves the arm.**
- **The MoveIt planning group is `fr3_arm`, not `fr3_manipulator`.** Only `fr3_arm` has a
  kinematics solver entry, which Humble's `computeCartesianPath` needs — `fr3_manipulator`
  returns `fraction=0.0` on every request *with* `error_code SUCCESS`, which reads like a
  planning failure and isn't one.
- **The Franka Hand cannot be servoed.** It is not in ros2_control at all, and libfranka offers
  only blocking `move`/`grasp`/`stop`. The server accepts every goal and preempts none, with new move commands
  going into a queue to be executed after each move completes.
- **Gripper width is `position[0] + position[1]`** on `/fr3_gripper/joint_states` — each finger
  reports half the aperture, and `velocity`/`effort` are hardcoded zero, so there is no force
  feedback to read.
- **The laptop and NUC run different rmw majors** (Humble 1.3.4 vs Kilted 4.0.2). Consequences:
  the `Failed to parse type hash` / `serdata.cpp` log noise is expected and unsuppressable;
  `ros2 node list` and `ros2 param get <nuc node>` come back **empty from the laptop** even
  though topics, TF and service calls all work, so never conclude a NUC node is missing from
  that; and large nested messages (a `MoveGroup.Goal`) arrive **corrupted**, which is why the
  MoveIt calls live on the NUC rather than on the laptop.
- **Discovery is unicast-only**, peers hardcoded to `10.0.0.1` / `10.0.0.2`. If the laptop is not
  actually on `10.0.0.1`, nothing finds anything and there is no multicast fallback.
- **The clocks must agree** or NUC-stamped TF lands outside the laptop's buffer
  ("extrapolation into the past"). The NUC's VLAN blocks outbound NTP, so its chrony syncs to the
  laptop over the arm link; check with `ssh jailfranka chronyc sources` → `^* 10.0.0.1`.
  `tf_use_latest:=true` is a stationary-dry-run crutch, never a fix.
- **The chunk has to outlast the latency budget**:
  `obs age + latency.<device>_exec < n_action_steps * action_dt`. If it doesn't, every action has
  already elapsed on arrival and *nothing moves at all* while every other indicator looks healthy.
  You can catch this by checking the latency monitor plot in Foxglove. Check [calibration-instructions.md](calibration-instructions.md), "Latencies", for more info on this latency calculation.
- **Two upstream `franka_ros2` v0.1.15 bugs, unfixed**: the gripper's params file is silently
  ignored (wrong node key, so `ros2 param dump /fr3_gripper` disagrees with the YAML), and there
  is no finger TF (`joint_state_publisher` subscribes to a topic nothing publishes). Both live in
  the NUC's `~/franka_ws`, not our submodule.

## Logs

`fr3_session.sh` tees the crash-prone launches to
`~/.local/state/polyumi/{fr3_bringup,fr3_inference,policy_client}_<date>.log` on whichever
machine ran them, since **`~/.ros/log/` will not have the lines you want after a crash** — `franka_bringup`
uses `output='screen'`, and a libfranka fault surfaces as raw stderr from `std::terminate`, not
rcl logging. The reflex name (`cartesian_reflex`, `joint_velocity_violation`, …) is in the Franka
Desk error log, which you must clear before bringup will come back.


## When it doesn't come up

Most failures here are one of four things, in rough order of frequency:

1. **A leftover launch** holding port 8765 and `/dev/video2` — `pkill -f "ros2 launch"`, then
   confirm a single stack before relaunching.
2. **The launching shell's DDS env** — an interactive rc that exports its own `ROS_DOMAIN_ID`
   overrides what tmux handed the pane. Check the live process, not your shell:
   `tr '\0' '\n' < /proc/$(pgrep -f policy_client_node)/environ | grep -E 'ROS_DOMAIN_ID|RMW_|CYCLONEDDS'`.
   Everything laptop-local keeps working, which is what makes this slow to spot.
3. **`fr3_bringup` died on the NUC** — `ros2 topic info /tf_static` shows `Publisher count: 0`.
4. **The arm stopped itself** — see Logs above.

## Gripper problems
Currently this setup uses the Franka Hand, which is terrible. I've done a bunch of analysis on it, tl;dr it has ~210ms observable command delay, only updates its state at 5Hz, and its move() commands cannot be pre-empted once issued. 

The scripts I used for this analysis are in `nuc/polyumi_fr3_controllers/src/franka_hand_testing`; I have a jupyter notebook to analyze the results as well--reach out to me if you need it for some reason.

The future of this system is to replace the Franka Hand with a better hand, as other labs have done, but for now I am working around its limitations as best I can with a custom interpolation scheme based on a model of the hand's response after all that testing.


Good luck, stranger!