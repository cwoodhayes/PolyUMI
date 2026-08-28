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

Safe commands run themselves; robot-moving ones are pre-typed for you to press Enter on. On
every fresh start it rsyncs `nuc/` to the NUC, runs `./deploy.sh` for the Pi and
`./deploy_lamb.sh` for lamb, so all three run this working copy (`SKIP_DEPLOY=1` skips it).

**lamb runs both the ROS client and the policy server.** This laptop is only a terminal, so it
can sleep mid-run: the NUC and lamb panes are tmux *on those hosts*, and re-running the script
re-attaches. Per-host link settings — NIC, static IP, CycloneDDS config — live in
`config/env.<hostname>.sh`, sourced by `setup_franka_env.sh`; lamb's are in
`config/env.lamb.sh`. `./fr3_session.sh --kill` tears the whole wall down cleanly (a plain pane
kill sends SIGHUP, which leaves the Pi's LED lit and the inference container running).

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
    execute_arm:=true execute_gripper:=true
```

**Both execute flags default false** — launching alone never moves the robot. 

Velocity scaling is only applicable if the arm is being controlled by moveit, which is deprecated aside from the homing functionality.

## 3. Inference server

`lamb` runs the policy server (port 8002) *and* the ROS client, so the client reaches it at
`http://localhost:8002` and the request never crosses the dedicated cable. `./deploy_lamb.sh`
pushes this working copy and runs the three build steps a plain rsync leaves stale;
`fr3_session.sh` calls it on every fresh start, so lamb runs what you have checked out.

The dummy and the real server are the **same app** — `create_app` in the `polyumi_inference`
library (`inference_server/`), which the ROS client imports too — with different backends. So a
frame the dummy accepts is one a checkpoint accepts, and a refusal you see during bringup is the
one you would have seen in production. If you change anything about the request, change it there:
both ends and both servers move together.

```bash
# real policy, on lamb (see training-instructions.md):
CKPT=/abs/path/to/<name>.ckpt ./serve_policy.sh
curl http://localhost:8002/health             # from lamb, where the client also runs

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

### Reading the round trip

`inference_latency_s` is the whole request, and on its own it cannot tell a busy GPU from a slow
link. The server reports its forward-pass time on the wire, so two more topics split it:

| Topic | What it is | What makes it grow |
|---|---|---|
| `inference_model_s` | The forward pass alone, timed through the server's `.cpu()` sync point | GPU work; whoever else is on that box |
| `inference_overhead_s` | The round trip minus the forward pass | Serialization and the link — scales with the observation payload |

The same split appears in the log line (`inference=NNNms = NN model + NN overhead`), and the
server's access log carries its own total alongside (`... in NNN ms, model NN ms`).

**Do not compute this from the server's total instead.** The server starts its clock in the
HTTP middleware but FastAPI reads the request body inside the endpoint, so a large upload is
still arriving while the server times itself — its total quietly absorbs link time. Measured
against a do-nothing echo server over this link, at the base64-encoded-uint8 stage this format
went through before the raw-frame rewrite, a 0.40 MB request reported 36 ms of "server" time on
a box doing nothing but a base64 decode — a cost the current raw-frame format doesn't have, since
it decodes nothing to get at the bytes. The forward pass is the only term measured cleanly on
either side, which is why the split hangs off it.

**The link is the thing to check first when `inference_overhead_s` is large.** The observation is
~0.30 MB of raw bytes (`n_obs_steps` frames of 224x224x3 uint8, no base64 — see `wire.py`), so the
wire time is set entirely by how fast the laptop can push that. The laptop's USB ethernet adapter
is an ASIX AX88772 — a USB 2.0 Fast Ethernet part, hard-capped at 100 Mbit, which puts a floor of
~24 ms under every inference. `cat /sys/class/net/<iface>/speed` says which side you are on; a
gigabit adapter takes that floor to ~2.4 ms and is the cheapest latency fix available.

Sending the frames as `uint8` rather than `float32`, and raw rather than base64, is what got the
payload down from 1.6 MB to 0.30 MB — the `/255` happens server-side in
`serve_obs.wire_to_obs_dict`, which is bit-identical, and the dataset stores `camera0_rgb` as
uint8 anyway. Don't widen it again on the way out.

Measured over the 100 Mbit link against a stdlib echo server (no model, so this is overhead
alone), 12 requests each on a warm connection, at the uint8-but-still-base64 stage this format
went through en route to the current raw frame:

| Request | Body | Overhead p50 |
|---|---|---|
| `float32` (before) | 1.61 MB | 196 ms |
| `uint8` base64 (intermediate) | 0.40 MB | 81 ms |

The current raw-frame format carries the same uint8 bytes with no base64 wrapper (~0.30 MB, a
further 4/3 cut) and was not independently re-benchmarked on this link — expect the overhead to
drop below the intermediate row's 81 ms by roughly that same factor, not to match it exactly.

## Architecture on the NUC

`fr3_bringup` owns the hardware: `franka_bringup` (controller_manager + the libfranka hardware
interface), the `fr3_arm_controller` spawner, `robot_state_publisher`, and
`static_transform_publisher`s for `polyumi_tcp` and for `fr3_link8 → fr3_hand`.

That second static TF is load-bearing. `load_gripper` now defaults **false**, because
`franka_hand_node` owns the libfranka gripper connection and only one process can. But
`franka.launch.py` feeds that one flag into `xacro hand:=` as well, so turning it off also drops
`fr3_hand` from `robot_description` — which would orphan `polyumi_tcp` and break the laptop's whole
observation lookup, with a symptom pointing nowhere near the flag. The static publisher fills the
hole; the constant lives in `nuc/tcp_calib.py` next to the TCP.

`fr3_inference` adds the three things that sit on top of it — move_group, `fr3_home_service`,
`franka_hand_node` — plus the `polyumi_cartesian_impedance_controller` spawner, spawned
`--inactive`. They start, fail and restart together without touching the arm's state.

The laptop publishes an entire inference **action chunk** (`n_action_steps` waypoints, not
`actions[0]`) as a `MultiDOFJointTrajectory`, which the NUC's `polyumi_cartesian_impedance_controller`
(1 kHz streaming servo) splices onto:

| topic | consumer |
|---|---|
| `/polyumi/target_poses_traj` | `polyumi_cartesian_impedance_controller` — 1 kHz streaming servo |
| `/polyumi/target_gripper` | `franka_hand_node` → libfranka `Gripper::move` |

A dead command path is loud rather than silent: nothing subscribes, the arm holds still, and the
client warns every second naming the topic it expected. The full contract is the module docstring
of `ros2_ws/src/polyumi_ros2/polyumi_ros2/target_chunk.py`.

`fr3_home_service` runs alongside it, serving only `/polyumi/home` — joint-space homing
through move_group. It and the streaming controller claim the same `<joint>/effort` interfaces, so
**exactly one holds the arm at a time** — `ros2 control list_controllers` tells you which, and
`/polyumi/home` swaps them itself around a home move:

```bash
# Arm must be STATIONARY: switching restarts the libfranka control loop.
ros2 control switch_controllers --deactivate fr3_arm_controller \
    --activate polyumi_cartesian_impedance_controller
```

(This is the manual form of the swap `/polyumi/home` does itself, both directions, hands back
afterwards.)

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
  only blocking `move`/`grasp`/`stop`. `stop()` does not pre-empt — it queues *behind* the running
  Move and costs the remainder of the stroke plus ~100 ms. So `franka_hand_node` runs one Move to
  completion at a time and chooses which setpoint to aim at, rather than trying to track them all.
- **Gripper width is `position[0] + position[1]`** on `/fr3_gripper/joint_states` — each finger
  reports half the aperture, and `velocity`/`effort` are hardcoded zero, so there is no force
  feedback to read.
- **The laptop and NUC run different rmw majors** (Humble 1.3.4 vs Kilted 4.0.2). Consequences:
  the `Failed to parse type hash` / `serdata.cpp` log noise is expected and unsuppressable;
  `ros2 node list` and `ros2 param get`/`param set` come back **empty from the laptop** even
  though topics, TF and service calls all work, so never conclude a NUC node is missing from
  that (to set a live parameter, call its `<node>/set_parameters` service directly); and large nested messages (a `MoveGroup.Goal`) arrive **corrupted**, which is why the
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
- **Two upstream `franka_ros2` v0.1.15 bugs.** The gripper's params file is silently ignored
  (wrong node key, so `ros2 param dump /fr3_gripper` disagrees with the YAML). And
  `joint_state_publisher`'s `source_list` names `franka_gripper/joint_states` while the gripper
  publishes on `fr3_gripper/joint_states`, so the fingers never reach `/joint_states`. Both are
  unfixed, and the second is invisible on the default path anyway: with `load_gripper:=false` the
  URDF has no hand, so `joint_state_publisher` discards `fr3_finger_joint1/2` as joints it does not
  know, whichever topic they arrive on. **Expect move_group to warn `complete state ... not yet
  known. Missing fr3_finger_joint1` forever** — it is pre-existing, it is not the gripper node
  failing, and there is still no finger TF. Fixing it needs a hand in `robot_description`.

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
5. **A wedged `ros2` CLI daemon on the NUC.** Signature is
   `xmlrpc.client.Fault: <Fault 1: "<class 'RuntimeError'>:!rclpy.ok()">` out of anything using
   `ros2 node`/`ros2 param`/`ros2 control`. Underneath it is the Humble↔Kilted rmw gap —
   `ros2 node list --no-daemon` gives the real error, `empty node name returned by the RMW layer` —
   which poisons the daemon's context, after which *everything* through it fails. Fix:
   **`ros2 daemon stop`** (it respawns). `ros2 topic list/echo/hz` and service calls are unaffected,
   which is what makes it slow to spot.

   Worth knowing because it used to break the arm silently: the launch's controller switch shelled
   out to `ros2 control switch_controllers`, which goes through the daemon, so a poisoned daemon
   left `fr3_arm_controller` holding the arm and the servo inactive while every pane looked fine —
   the arm just never moved. That call is now a daemon-free `ros2 service call`, but if you see the
   fault anywhere else, this is it.

## Gripper problems
Currently this setup uses the Franka Hand, which is terrible. I've done a bunch of analysis on it, tl;dr it has ~210ms observable command delay, only updates its state at 5Hz, and its move() commands cannot be pre-empted once issued.

The scripts I used for this analysis are in `nuc/polyumi_fr3_controllers/src/franka_hand_testing`, gated off the default build behind `-DBUILD_HAND_PROBES=ON`. The constants below were fitted to their output in a Jupyter notebook that is not in the repo; the values as shipped live in `HandLimits` (`gripper_trajectory_interpolator.hpp`), pinned by the anchor tests in `test_gripper_trajectory_interpolator.cpp`. Re-run the probes against any other hand before trusting them.

`franka_hand_node` works around all of this with a custom interpolator built on that model:

```
blocked = C + duration(dx, v)                  # send -> move() returns
duration = dx/v + v/a       if dx >= v^2/a     # trapezoidal
         = 2*sqrt(dx/a)     otherwise          # triangular, never reaches v
v <- min(v_cmd, V_MAX)                         # the hand clips silently, it never refuses
```

| constant | value | meaning |
|---|---|---|
| `cmd_delay` | 0.208 s | send → fingers start moving, as *observed* |
| `C` (`fixed_cost`) | 0.363 s | send → `move()` returns, at zero travel |
| `V_MAX` | 0.1153 m/s | where the hand starts silently clipping |
| `A_MAX` | 0.360 m/s² | sets the 37 mm triangular crossover |
| `t_obs_delay` | 0.050 s | **a guess**: how far the reported width lags reality |

The node holds the chunk as a horizon of absolutely-timed widths and, for each Move, picks the
earliest setpoint it can still reach and the *slowest* speed that lands on time. When nothing is
reachable it chases full-speed to where the signal will be on arrival, rather than stopping short
— covering the remainder afterwards would cost another whole `C`.

Things to know before you debug it:

- **It is a decimator, not a tracker.** `blockedDuration(0) = 363 ms`, so the ceiling is 2.75 Hz
  and realistically 0.7–1.7 Hz against a 10 Hz setpoint stream. Servicing every 4th–15th setpoint
  is correct behaviour.
- **Transients shorter than ~1 s are physically unrepresentable.** A close-and-reopen inside 0.7 s
  cannot be rendered at all; two strokes plus their two `C`s exceed the transient.
- **`t_obs_delay_s` is the knob if the hand is consistently early or late** under
  `latency_probe -p mode:=gripper_chirp`. It cannot be measured without a camera on the fingers,
  which is why it is a parameter and not a constant. Do NOT confuse it with `latency.gripper_exec`
  on the laptop — that aligns the action chunk, this shifts the node's internal schedule.
- **`Grasp` is not implemented.** `Move` applies no force and stalls on contact, so **the hand
  cannot hold an object**. The node warns when a successful Move reports a width the hand did not
  reach, which is what that failure looks like.
- **An unhomed hand reports `max_width = 0` and `move()` returns `true` while doing nothing.** The
  node refuses to execute in that state and says so; `home_on_start:=true` fixes it.

The future of this system is to replace the Franka Hand with a better hand, as other labs have
done. `franka_hand_node` is the stopgap until then, and it has not yet been run on the arm:

- [ ] **On-arm dry run.** `execute_gripper:=false` (the default) plans and logs every
      `move(width, speed)` at the real cadence without connecting to the hand.
- [ ] **On-arm execution.** `execute_gripper:=true`, arm plan-only. Acceptance test is
      `ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=gripper_chirp`.
- [ ] **Confirm or replace `latency.gripper_exec`.** Shipped at **0.0, under test** — the node now
      schedules each Move to arrive on time by itself, so a lead here would double-compensate.
      Revert to 0.380 if the hand runs late in service.
- [ ] **`Grasp`**, if the closed endpoint has to be force-defined. See `gripper_range_probe`, which
      fails when the fingers do not stall repeatably.

Re-run the probes (`-DBUILD_HAND_PROBES=ON`) against any candidate replacement before committing to
it — the constants above are this hand's, and nothing else in the stack will notice if they are
wrong.


Good luck, stranger!