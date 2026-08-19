# Franka Inference Bringup — plan & progress

**Scope: planning and progress tracking, not reference documentation.** When something here is
finished, the truth moves into the code and its docstrings, and the entry here gets deleted rather
than annotated. Reach for these instead:

| For | Read |
|---|---|
| How to run the stack, and what breaks | [crb-fr3-inference.md](crb-fr3-inference.md) |
| Pose/gripper/image data conventions | [data-format.md](data-format.md) |
| Measuring the constants | [calibration-instructions.md](calibration-instructions.md) |
| TCP frame geometry | `nuc/tcp_calib.py` |
| Gripper width units | `ros2_ws/src/polyumi_ros2/polyumi_ros2/gripper_map.py` |
| Wire contract | `inference_server/`, `serve_obs.py` + their tests |

---

## Status

The pose/vision path is structurally complete end to end: latency compensation and frame
conventions follow UMI, and single action chunks move the real arm. Every latency the inference
path consumes is now measured. What remains is three unwired signals, the executor redesign, and
the training side.

| Workstream | State |
|---|---|
| FR3 NUC ↔ laptop over DDS | **done** |
| Dummy server + client round trip | **done** |
| Action-chunk execution on hardware | **done** for single chunks; continuous 10 Hz loop unverified. Executor redesign is Phase 4 |
| Latency compensation, gopro + proprio | **done**, matches UMI's scheme, unit-tested. Values measured 2026-08-10/11 — `gopro` 0.1087, `arm_exec` 0.620, `gripper_exec` 0.514, `gripper` 0.0214 |
| Latency compensation, finger cam + piezo | **not started** — params declared, never consumed |
| Pose body frame (training ↔ inference) | **done, verified on hardware** — `polyumi_tcp` end to end, CAD-measured, confirmed by `tcp_pivot_test` |
| Camera pixel transform (training ↔ inference) | **done** — shared crop+resize contract, pinned by cross-environment golden digests |
| Gripper obs + command | **done** — `/fr3_gripper/joint_states` → `agent_pos[7]`; `action[7]` → `/polyumi/target_gripper` → NUC bridge |
| Gripper width calibration | **done, measured 2026-08-09** — closed width 44.56 mm, closed aperture 0.0 m, open aperture 0.0816 m |
| Receding-horizon inference stride | **done** — `steps_per_inference` (default 6) |
| DP export | **works**; UMI schema + tests landed. Still no tactile, and the rework in "What's left" is outstanding |
| Real inference server | **in progress** — `serve_policy.py` green standalone on sheep; client wiring done + unit-tested; on-arm dry run pending hardware |

---

## What's left

3. **Finger cam + piezo are unwired.** Params exist and are never consumed. If they become
   observations, the capture instant becomes the *oldest* across streams — an observation is only
   as fresh as its slowest signal — and they must be added to the DP export, which carries neither.

4. **DP exporter rework.** Deferred as one chunk rather than patched piecemeal:
   - **Hard-requires OptiTrack even for SLAM-sourced poses.** `_export_episode` reads
     `optitrack/timestamps` unconditionally to clip the overlap window, so a SLAM-only scene
     raises `KeyError` *after* step 5 wrote a perfectly good `eef/pose`. The window should clip to
     the sources the episode actually uses.
   - **No tactile.** Piezo audio and finger-camera frames aren't exported; the exporter touches
     `timestamps/finger` only to compute the window.
   - **The GoPro→finger clock shift is duplicated** between `buffer.py` (inline) and
     `eef_pose_step._gopro_ts_in_finger_clock`, and they disagree on strictness: the exporter
     requires `annotations/time_sync`, the step defaults the offset to `0.0`. Picking one is a
     behaviour change that belongs with this rework.

   `ingest/test/test_dp_export.py` covers the file now (schema, segmentation, pose-source
   resolution, quality gating, width convention) — extend it as part of the rework.

5. **Phase 4** — the executor redesign, below.

---

## Hard-won gotchas

Things that cost real time to discover and would plausibly be re-attempted. Everything else that
was once in this doc has been deleted.

### Cross-machine (laptop Kilted ↔ NUC Humble)

- **MoveIt goals do not survive the rmw-major gap.** A `MoveGroup.Goal` / `GetCartesianPath.Request`
  sent laptop→NUC arrives corrupted (move_group logs `Catastrophic failure`). This is why the
  MoveIt calls live in `nuc/fr3_moveit_bridge.py` **on the NUC**, same rmw as move_group.
- **Small flat messages cross fine.** `PoseArray`, `trajectory_msgs/JointTrajectory` and
  `trajectory_msgs/MultiDOFJointTrajectory` were each verified byte-for-byte across the gap —
  the last including per-point `transforms` and `time_from_start`, which is what the streaming
  executor rides on. Service calls also work — `gripper_range_probe` queries the NUC bridge's
  parameter service from the laptop.
- **`moveit_py` cannot be a thin client** to the NUC's move_group: it needs
  `robot_description`/SRDF in-process, which needs the Humble-only `franka_description`. Raw
  `moveit_msgs` calls instead.

### MoveIt / arm

- **Plan against SRDF group `fr3_arm`, not `fr3_manipulator`.** Only `fr3_arm` has an IK solver
  entry in `kinematics.yaml`; `fr3_manipulator` returns `fraction=0.0` for every Cartesian request
  on this Humble MoveIt.
- **Humble's `GetCartesianPath` has no velocity-scaling field** (added later upstream); the bridge
  time-scales the planned trajectory instead.
- **`franka_fr3_moveit_config`'s `move_group.launch.py` does not work as shipped** — undeclared
  launch args, and missing the OMPL/controller/planning-scene-monitor params move_group needs to
  execute. `nuc/launch/fr3_move_group.launch.py` is the fixed copy.

### Frames

- **`O_T_EE` is `fr3_hand_tcp`** (verified 2026-08-19): `/franka_robot_state_broadcaster/current_pose`
  and TF `fr3_link0→fr3_hand_tcp` agree to the printed digits, while `fr3_link8` and `fr3_hand` do
  not. Anything reading `O_T_EE` or a Jacobian at `franka::Frame::kEndEffector` is working at that
  frame, not at `polyumi_tcp`, and owes the difference a transform.
- **`franka_msgs/srv/SetTCPFrame` is not the lever** for the TCP. It only changes libfranka's
  `O_T_EE` reporting; TF and MoveIt are driven entirely by the URDF. It *would* move the control
  point of a torque controller reading `O_T_EE` — but then `O_T_EE` and TF disagree about where the
  EE is for every other consumer, so the streaming controller does the transform itself instead.
- **Nothing in software can validate the TCP.** TF always reports it exactly where the URDF says,
  so the model-vs-reality gap is only visible in the room — `ros2 run polyumi_ros2 tcp_pivot_test`.

### Policy / serving

- **Use `ema_model`, not `model`.** EMA weights are what eval uses; `.model` runs but is worse.
- **Checkpoints are dill-pickled against the exact dep tree**, and normalization ships baked into
  the state dict. Serving and training must use the same image. A wrong action parameterization is
  likewise baked in permanently.
- **UMI's policy returns the full horizon with no offset** (`diffusion_unet_timm_policy` has no
  `n_action_steps`; read `result['action_pred'][0]`). Truncation is the client's job.
- **Rel→abs is `convert_pose_mat_rep(..., backward=True)`** with `pose_rep='relative'`, i.e.
  `base_pose_mat @ pose_mat`. Don't re-derive it.

### Gripper hardware

- **The Franka Hand cannot be servoed. This is a hardware ceiling, not an API gap** — worth stating
  because the obvious instinct is to fold the gripper into Phase 4's streaming redesign.
  `franka::Gripper` (libfranka `gripper.h`) exposes exactly `homing`/`grasp`/`move`/`stop`/
  `readOnce`, all blocking and discrete, and `ros2 control list_hardware_interfaces` shows the hand
  has **zero** ros2_control interfaces — no analogue of the arm's `cartesian_pose`. UMI servos its
  WSG50 at 30 Hz (`wsg_controller.py:144-250`) around the same interpolator as its arm; we cannot.
- **It must also be rate-limited.** `gripper_action_server.cpp` returns `ACCEPT_AND_EXECUTE`
  unconditionally and spawns a detached thread per goal — no queue, no preemption. libfranka aborts
  the superseded command. Streaming at 10 Hz would mean ~9 spurious `Command aborted!` per second
  plus unbounded thread churn. Hence a discrete, deadbanded, rate-limited commander, and hence a
  **separate node** from `fr3_moveit_bridge`, whose `_busy` lock would otherwise drop gripper
  commands precisely while an arm chunk executes.
- **`Move` applies no force** and stalls on contact; `Grasp` is the only action that *holds* an
  object, opt-in via `use_grasp_below_m`, shipped disabled.
- **`franka_gripper` publishes `max_width` on no topic**, so neither end of the range is readable
  at runtime — both are measured constants.

---

## Phase 4 — streaming Cartesian impedance servo (written; on-arm bringup pending)

**Plan of record for closing the inference-latency / jerky-motion gap.** Traced from upstream UMI
(`../universal_manipulation_interface`), `../polymetis`, `../serl_franka_controllers`, and live
probes of the NUC.

### Why

Our on-arm path treats camera + inference latency as a *failure* (drop the tick / drop the chunk)
and drives the arm **plan-then-execute**, position-controlled, with no compliance. UMI treats that
latency as an expected, compensated quantity and drives the arm with a **continuous 1 kHz Cartesian
impedance servo**. This is architectural, not a tuning constant: plan-then-execute starts each chunk
from rest, stops at its end, and discards the policy's intended `dt` timeline. And the tasks involve
contact, which a stiff joint-trajectory controller handles badly.

UMI's executor (`franka_interpolation_controller.py:277-355`) is a 1 kHz loop around a
`PoseTrajectoryInterpolator` (`umi/common/pose_trajectory_interpolator.py`, pure numpy). Every 1 ms
it evaluates the interpolator and writes the desired EE pose. New chunks arrive as
`schedule_waypoint(pose, target_time)`, splicing each future waypoint in at its absolute time with
no stop. Action chunks are timestamped in absolute wall-clock anchored to the observation
(`action_timestamps = arange(len)*dt + obs_timestamps[-1]`, `eval_real.py:503`); in-past waypoints
are dropped.

The design splits cleanly in two, and the halves come from different places:

| Layer | Source | Job |
|---|---|---|
| Reference generator | UMI `PoseTrajectoryInterpolator` | 10 Hz chunk with absolute timestamps → continuous 1 kHz equilibrium pose |
| Control law | SERL `CartesianImpedanceController` | (equilibrium pose, measured state) → joint torques |

### Remaining gaps

| # | Gap | UMI | Ours | Fix |
|---|-----|-----|------|-----|
| 1 | **Executor model** | 1 kHz impedance servo, splices chunks | `compute_cartesian_path` + blocking `ExecuteTrajectory`, skip-while-busy | ros2_control torque controller around a ported `PoseTrajectoryInterpolator` |
| 2 | **Action timing** | absolute wall-clock per waypoint | `PoseArray`, no per-waypoint time; NUC re-times the chunk | Carry per-waypoint absolute times in a `MultiDOFJointTrajectory` |

Inference cadence was the third gap and is **closed** — `steps_per_inference` (default 6) already
strides the laptop side. Gap 2 needs both machines to agree on wall-clock time, which the chrony
sync gives us (CLAUDE.md, "Clock sync").

### Why torque control, not the `cartesian_pose` motion generator

`franka_hardware` exposes a native Cartesian-pose command interface (16 doubles, column-major
`O_T_EE`), which looks like the direct analogue of UMI's `update_desired_ee_pose`. It is not enough.
Under it, `Robot::initializeCartesianPoseInterface()` hardcodes
`startCartesianPoseControl(ControllerMode::kJointImpedance)` (`robot.cpp:294`), so the arm is
position-commanded and tracked by Franka's internal **joint**-impedance controller. Its gains are
stiffness-only — `setJointImpedance(K_theta[7])`, `setCartesianImpedance(K_x[6])`, both reachable
via `/service_server/set_{joint,cartesian}_stiffness` — and **there is no damping knob at all**.
Contact tasks need a real Cartesian mass-spring-damper, so we command joint torques and own the law.

### Where the law comes from

Polymetis (what UMI actually runs) computes, in `torchcontrol/modules/feedback.py`
(`OperationalSpacePD.forward`) plus `policies/ee_pd.py`:

```
tau = J^T (Kp·e + Kd·(−J·dq)) + coriolis        # InverseDynamics(..., ignore_gravity=True)
```

`serl_franka_controllers/src/cartesian_impedance_controller.cpp` is the same law with the opposite
error sign convention, plus three things: an **error clip** (their contribution — high stiffness for
accuracy, bounded force on contact), a **nullspace term**, and an optional integral term defaulting
to 0. Polymetis has no nullspace control at all, which on a 7-DOF arm lets the elbow drift; SERL's
addition is an improvement to keep, not a divergence to undo.

### NUC capability findings (live probes, latest 2026-08-19)

Expensive to re-derive, so kept:

- **`controller_manager` runs at 1000 Hz** (`franka_bringup/config/controllers.yaml`).
- **The NUC has a realtime kernel**: `5.15.0-1103-realtime`, `PREEMPT_RT`, `@realtime` at
  `rtprio 99`. Torque control is viable here; it would not be on the laptop.
- **Everything the law needs exists in franka_ros2 v0.1.15.**
  `franka_semantic_components::FrankaRobotModel` gives `getZeroJacobian(frame)`,
  `getCoriolisForceVector()`, `getMassMatrix()`; `tau_J_d` rides on
  `FrankaRobotState.desired_joint_state.effort`.
  `franka_example_controllers/src/joint_impedance_with_ik_example_controller.cpp` is a working ROS 2
  skeleton claiming `<joint>/effort` plus the robot-model state interfaces.
- **`franka_semantic_components` is installed** on the NUC — that is what we link.
  `franka_example_controllers` is present in `~/franka_ws/src/franka_ros2/` but unbuilt; we only
  read it as a template, so it does not need building.
- **`moveit_servo` is NOT installed** — the ROS-native servo shortcut is out unless added.
- `franka_hardware` applies **no** continuity safety net: `cartesian_pose_low_pass_filter_active_`
  and `cartesian_pose_command_rate_limit_active_` are hardcoded `false` (`robot.hpp:321-322`) with
  no setter.

### Staged design

**Stage 1 — laptop, low-risk.** Publish a **timestamped trajectory** alongside the `PoseArray`:
`trajectory_msgs/MultiDOFJointTrajectory` on `/polyumi/target_poses_traj`, per-point `transform` +
`time_from_start`, `header.stamp = t_obs - latency.arm_exec`. Carries UMI's
`(poses, action_timestamps)` exactly, with the per-waypoint latency subtraction folded into the
anchor. Keep the client-side coarse stale-drop; the NUC does fine scheduling.

**Stage 2 — NUC, the real work.** New ament_cmake package `nuc/polyumi_fr3_controllers/`:
port `PoseTrajectoryInterpolator` to Eigen; port SERL's law as free functions (so it is testable
without hardware); wrap both in a ros2_control controller claiming `<joint>/effort`. It seeds the
interpolator and `q_d_nullspace` from the current state on activation, subscribes to the Stage-1
topic and `schedule_waypoint`s each future pose, and in `update()` at 1 kHz evaluates
`pose_interp(now)` as the equilibrium pose. SERL's first-order lag on the equilibrium pose is
dropped — the interpolator already guarantees continuity and the lag would only add tracking delay.
This retires `fr3_moveit_bridge` from the inference path; it stays for homing.

**Stage 3 — gains + latency.** Start from SERL's defaults, not UMI's: 2000 N/m translational with a
±0.01 m clip bounds commanded force near 20 N while keeping tracking accuracy, where UMI's softer
750 N/m spring is unbounded. Re-measure `latency.arm_exec` — it stops being a MoveIt planning time
and becomes transport plus one control cycle.

### Risks

- **libfranka continuity limits.** The interpolator is mandatory — never step the equilibrium
  point — and activation MUST seed from the current pose so the first command is a zero-error one.
  Nothing in `franka_hardware` will catch a discontinuity for us.
- **We own stability.** Under torque control a bug throws the arm, where the `cartesian_pose` route
  would have had the firmware guaranteeing it. `saturateTorqueRate` (1.0 Nm/cycle) and the error
  clip are the two bounds that make this tractable; neither is optional.
- **Collision thresholds abort intentional contact.** `DefaultRobotBehavior`'s defaults are tuned
  for "contact means something went wrong". Raise them via `/service_server/set_full_collision_behavior`.
- **The control point is `polyumi_tcp`, ~15 cm from `O_T_EE`.** A spring anchored at the EE gives
  visibly wrong compliance at the fingertips, so the Jacobian needs the adjoint shift.
- **The gripper cannot join this redesign** — see the hardware ceiling above. It keeps its own
  channel and its own discrete commander. `serl_franka_controllers` has no gripper code either;
  SERL drives the hand out-of-band through stock `franka_gripper`, exactly as we do.

### Checklist

Stages 1 and 2 are written and green off-hardware. What is left is all on the arm — see
[Phase 4 bringup](#phase-4-bringup-the-on-arm-sequence) at the end of this doc.


## Next hardware session

Everything below is blocked only on arm/GoPro access.

- [ ] **Arm dry run, no execution** (`execute_motion:=false`, the default).
      1. Server on sheep: `CKPT=/abs/path/to/<name>.ckpt ./serve_policy.sh`; from the laptop
         `curl http://<sheep-ip>:8000/health` → `ready`.
      2. Laptop: `source setup_franka_env.sh`, NUC publishing `fr3_*` TF, GoPro streaming, then
         `ros2 launch polyumi_ros2 inference_demo.launch.xml inference_server_url:=http://<sheep-ip>:8000/predict_cartesian/`
      3. Watch: node logs `mode: log-only (no motion)`, one `/reset`, then per-tick chunk logs;
         `/health` shows `episode_start_set: true`. In Foxglove (`:8765`) add
         `/polyumi/target_poses_preview` — the chunk should sit near the current EEF and step
         smoothly. Wild jumps, NaNs, or off-workspace poses are the finding.
      4. Also confirm the startup line `camera crop: 1920x1080 → 1440x1080, discarded bars mean
         …/255 — pillarbox as expected`. An error there means the crop is eating real image.
- [ ] **Gripper on-arm dry run** (`fr3_gripper_bridge` with `execute:=false`)
- [ ] **Gripper on-arm execution** (`execute:=true`, arm bridge plan-only)

Anything that **moves the arm** now lives in [Phase 4 bringup](#phase-4-bringup-the-on-arm-sequence)
instead. Arm execution and the continuous 10 Hz loop were listed here against the MoveIt executor,
which the streaming controller replaces — testing that path end-to-end would be work spent on
something step 9 deletes. The dry run above is still worth doing first either way: it exercises the
whole observation and inference round trip without commanding motion, so it isolates the policy from
the executor.

---

## Still-open questions

| # | Question | Status |
|---|---|---|
| 1 | Do finger cam / piezo feed the policy, and at what latency? | **Open.** Params exist, unconsumed. See "What's left" 3. |

---

## Phase 4 bringup: the on-arm sequence

**This is the live worklist.** Tick items as they pass; delete them once the whole section is done
and the mechanism has moved into [crb-fr3-inference.md](crb-fr3-inference.md) and the controller's
own docstring, per this doc's scope rule.

The steps are ordered so that each one leaves exactly one new thing unproven. This is a **torque
controller**: unlike the `cartesian_pose` route, no firmware guarantees stability, so a step that
looks boring is a step doing its job. Do not skip ahead to the policy.

### What already holds

Verified off-hardware, so a failure below is not these:

| | |
|---|---|
| `PoseTrajectoryInterpolator` port | 8 gtests, expected values generated by running UMI's own Python on the same inputs |
| Cartesian impedance law | 13 gtests — zero-error/zero-torque, force opposes error, clip saturates, nullspace ⊥ task, rate bound |
| `colcon test` on the NUC | 23/23, Release build clean |
| `policy_client_node` MultiDOF output | 3 gtests — pre-slice indexing, `arm_exec`-anchored stamp, frames |
| `/polyumi/home` handover | 4 tests — both directions, STRICT, hands back on planning failure |

The NUC is already set up: `~/franka_ws/src/polyumi_fr3_controllers` is symlinked to the repo and
built. After editing C++, `rsync` then rebuild:

```bash
# laptop
rsync -a --exclude='__pycache__/' nuc jailfranka:~/Documents/PolyUMI/
# NUC
cd ~/franka_ws && colcon build --packages-select polyumi_fr3_controllers \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
```

**Then restart `fr3_bringup`.** pluginlib keeps a library mapped once it has dlopen'd it, so a
`controller_manager` that has already loaded (or tried to load) the controller keeps running the
old `.so` even after the file on disk is replaced. Skipping the restart gives you the *previous*
build's behaviour — including its error messages — which is indistinguishable from the fix not
working.

### Prerequisites

`fr3_bringup` is required for **every** step below — it owns the `controller_manager` the
controller loads into, and publishes the `polyumi_tcp` static TF the controller looks up on
activation. Enable FCI on the Desk UI first.

| Steps | Stack |
|---|---|
| 1–2 | `fr3_bringup` alone, plus a manual spawn (below) |
| 3 | `fr3_bringup`; blocked on tooling |
| 4–7 | `fr3_bringup` + `fr3_inference` (for move_group and the bridges) |

Steps 1–2 deliberately do **not** use `fr3_inference` — it would start move_group and both bridges
for nothing. Keep the first torque-control activation to the smallest stack that can express it:

```bash
# NUC, after fr3_bringup
ros2 run controller_manager spawner polyumi_cartesian_impedance_controller \
  -t polyumi_fr3_controllers/CartesianImpedanceController \
  --param-file ~/Documents/PolyUMI/nuc/config/polyumi_controllers.yaml --inactive
```

`-t` is **not** optional and not redundant with `--param-file`: Humble's spawner sets the `type`
param only from `-t` (`spawner.py:208`), while `--param-file` sets the controller's `params_file`
and never reads a type out of it. Without it the manager rejects the load with *"The 'type' param
was not defined for ..."*, which reads like a problem with the controller rather than a missing
flag. The reason is worth knowing because the failure is logged by the **controller_manager**, not
the spawner — the spawner only says `Failed loading controller`, so the actual cause is in
`~/.local/state/polyumi/fr3_bringup_<date>.log`.

### The sequence

- [x] **1. Activate with no target published.** Arm **stationary** — switching restarts the
      libfranka control loop (`perform_command_mode_switch` calls `stopRobot()` then
      re-initialises). **Passed 2026-08-19**, first try: activated clean, held position, no jump.

      ```bash
      ros2 control switch_controllers \
          --deactivate fr3_arm_controller \
          --activate polyumi_cartesian_impedance_controller
      ```

      **Pass:** the arm holds position dead still and the controller logs the pose it is holding
      plus the offset it resolved, which should read
      `fr3_hand_tcp -> polyumi_tcp offset (0.0196, -0.0000, 0.1535)` — confirmed on hardware
      2026-08-19. That is `polyumi_tcp` expressed *in* `fr3_hand_tcp`, i.e. `tcp_calib.py`'s
      `(0.019612, 0, 0.2569)` with franka_description's 0.1034 m `fr3_hand -> fr3_hand_tcp` taken
      off the z. Note it is **not** the inverse of that; `lookupTransform(A, B)` returns B in A.

      **A jump on activation means the TCP lookup or the Jacobian shift is wrong — stop.** Do not
      tune around it. The controller seeds its equilibrium from the measured pose precisely so that
      error is zero on the first `update()`; if the arm moves, the pose it thinks it is at is not
      the pose it is at, and every subsequent step inherits that error. A ~15 cm jump specifically
      points at the `polyumi_tcp` transform.

- [x] **2. Push it by hand.** The cheapest complete test of the control law, and it needs no
      trajectory at all.

      **Pass:** it springs back, and the resisting force stops growing once you are past the error
      clip — roughly 20 N at the shipped gains (2000 N/m × 0.01 m). Then, live:

      ```bash
      # NOT `ros2 param set` — see below.
      ros2 service call /polyumi_cartesian_impedance_controller/set_parameters \
        rcl_interfaces/srv/SetParameters \
        "{parameters: [{name: 'translational_stiffness', value: {type: 3, double_value: 500.0}}]}"
      ```

      and confirm it gets noticeably softer within ~0.5 s. That exercises the full path: state
      read, TCP transform, error, clip, gains, torque, saturation. Every gain, clip and speed limit
      is live-tunable this way; the controller re-reads them off the realtime thread at 2 Hz, so
      tuning against a real contact does not need a deactivate/activate cycle (which would stop and
      restart the libfranka loop between every attempt).

      **`ros2 param set` does not work here**, and the traceback does not say why:

      ```
      RCLError: Failed to get node names: empty node name returned by the RMW layer
      ```

      That is the rmw-version gap, described in
      [crb-fr3-inference.md](crb-fr3-inference.md) — the ROS *graph* does not cross Humble↔Kilted,
      so node-name enumeration fails on the NUC whenever the laptop's participants are on the same
      domain. `ros2 param set` calls `get_node_names()` to validate the node before setting it, so
      it dies there rather than on anything to do with the parameter. Service calls match on DDS
      endpoints instead of the graph and are unaffected, which is why the form above works. The same
      substitution applies to any `ros2 param`/`ros2 node` command in this setup.

      Watch for **oscillation** rather than a clean spring — that is the damping being wrong for
      the stiffness, not the clip. Rotational damping ships deliberately under-damped (150/7
      against a critical ~24.5); if rotation rings, that is the first knob.

      **Passed 2026-08-19**: bounded spring, and the live stiffness sweep softens it as expected.

      Getting there took two fixes, both since landed, and both worth knowing about because they
      failed invisibly rather than with wrong numbers. The first push tripped `cartesian_reflex`
      and took `ros2_control_node` down with it (exit -6) — franka's Cartesian collision reflex
      fires on estimated *external* force, and a hand push is external force by definition, so the
      controller now applies thresholds itself (see "Collision thresholds"). And the gains were
      read only at configure and activate, so the sweep could not have changed anything even once
      `set_parameters` reached the node.

      > Everything below this line is blocked on the probe tooling. Stop here for the first
      > session and confirm 1–2 before building it.

- [x] **3. Synthetic slow trajectory, no policy.** **Passed 2026-08-19.**

      ```bash
      # laptop, with the impedance controller ACTIVE on the NUC
      ros2 run polyumi_ros2 servo_smoke_test
      ```

      A 3 cm circle around wherever the TCP already is, 12 s per lap, driven by **overlapping**
      16-waypoint chunks at ~3.3 Hz — the same traffic shape the policy produces. That overlap is
      the point: it is the only test here where two multi-waypoint chunks are in flight at once,
      which is what the interpolator's splice exists for. `tcp_pivot_test` publishes one chunk and
      waits; `latency_probe` publishes single waypoints; neither exercises it.

      **Pass:** smooth continuous motion, no pause at chunk boundaries, no torque chatter, no
      `cartesian_reflex`. A stutter at exactly `chunk_hz` is the splice, not the gains. Bisect
      `max_pos_speed` down if it faults.

- [ ] **4. `/polyumi/home` round trip.** `ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"`.
      **Pass:** it switches out, homes, switches back, and the arm holds still afterwards.
      `ros2 control list_controllers` should show the impedance controller `active` again.

- [ ] **5. `tcp_pivot_test` through the impedance controller.** It now defaults to the timed wire
      format, so no extra flag; `-p wire:=pose_array` still drives the MoveIt path.
      **Pass:** closed fingertips stay visibly still. Same check that validated the TCP under
      MoveIt; here it additionally validates the Jacobian shift, which nothing else does in the
      room.

- [ ] **6. End-to-end with the policy.** The finding to watch for is not accuracy, it is
      **continuity across chunk boundaries** — no stop-and-go at 10 Hz. Compare the motion against
      `/polyumi/target_poses_preview` in Foxglove.

      If the arm holds still while everything else looks healthy, the two failure modes and how to
      tell them apart are in [crb-fr3-inference.md](crb-fr3-inference.md) Troubleshooting
      ("the arm holds still…", both entries).

- [ ] **7. Deliberate contact.** Drive into a fixed surface.
      **Pass:** the arm complies and does not trip Franka's collision monitor. If it does, raise the
      thresholds via `/service_server/set_full_collision_behavior` — the defaults treat contact as a
      fault, which is wrong for these tasks. The controller deliberately does not set them itself.

- [ ] **8. Re-measure `latency.arm_exec`** with `ros2 run polyumi_ros2 latency_probe --ros-args
      -p mode:=arm`, which also defaults to the timed format now. Note this is a genuinely
      different quantity from the old figure, not a better measurement of the same one: transport
      plus a control cycle instead of a MoveIt planning time. It also does two jobs under the servo
      — it still feeds `_n_stale_actions`, and it is subtracted from every chunk anchor, so an
      error here shifts the whole commanded timeline rather than just rounding a drop count.
      Once it lands:
      - `ros2_ws/src/polyumi_ros2/config/inference.yaml` — re-check whether `n_action_steps: 16` is
        still needed (it was forced up by the 0.62 s MoveIt budget), and drop the "unacceptably
        slow; must migrate to streaming controller" comment.
      - `docs/calibration-instructions.md` — rewrite Latencies gotcha 1 ("not a transport constant
        today… when the Phase 4 streaming controller lands") in the present tense, and gotcha 2
        ("the arm cannot be driven broadband… MoveIt's cadence caps how fast the arm can be swept"),
        which the servo removes.

- [ ] **9. Retire the MoveIt executor from the inference path.** Only after 6 passes. Delete
      `_on_target`, `_plan_cartesian`, `_run_execute` and the `PoseArray` subscription from
      `nuc/fr3_moveit_bridge.py`, and the `PoseArray` command publisher (not the preview) from
      `policy_client_node.py`. The node stays for `/polyumi/home`.

- [ ] **10. Configure the end-effector load.** Deferred deliberately — see "End-effector load"
      below. Do it programmatically (`franka_msgs/srv/SetLoad`) rather than in the Desk UI, so the
      value lives in the repo next to the gains rather than in a robot's settings.

### Collision thresholds

`cartesian_reflex` aborts the motion **and kills `ros2_control_node`**, so it is not a soft failure
you can push through. Raise the thresholds before any step that involves contact — which includes
step 2, since a hand push is contact:

```bash
# NUC. franka_example_controllers' own DefaultRobotBehavior values; every franka example
# controller applies these in on_configure. Nominal and acceleration are set to the same numbers.
ros2 service call /service_server/set_full_collision_behavior \
  franka_msgs/srv/SetFullCollisionBehavior \
  "{lower_torque_thresholds_nominal:      [25,25,22,20,19,17,14],
    upper_torque_thresholds_nominal:      [35,35,32,30,29,27,24],
    lower_torque_thresholds_acceleration: [25,25,22,20,19,17,14],
    upper_torque_thresholds_acceleration: [35,35,32,30,29,27,24],
    lower_force_thresholds_nominal:       [30,30,30,25,25,25],
    upper_force_thresholds_nominal:       [40,40,40,35,35,35],
    lower_force_thresholds_acceleration:  [30,30,30,25,25,25],
    upper_force_thresholds_acceleration:  [40,40,40,35,35,35]}"
```

These are franka's own defaults for the examples, not an aggressive setting — the point is that
*nothing* currently applies them, so the arm is running on the firmware's more conservative
factory values. Step 7 (deliberate contact against a surface) will likely need them higher still.

**The controller applies these itself at `on_configure`**, from the `collision.*` parameters in
`polyumi_controllers.yaml`, and **fails to configure if they do not take**. That is deliberate:
`set_full_collision_behavior` is a non-realtime command whose effect is invisible in TF, the topic
list and the controller's own logs, so a session where it silently did not apply is
indistinguishable from one where the gains are wrong. Refusing to load is the only honest signal.
Look for `collision thresholds set — force upper (...) N / (...) Nm` in the log. The command above
remains the way to try a value before committing it to the yaml.

### End-effector load (deferred)

The firmware's gravity compensation uses the **load mass configured on the robot**, and nothing in
this repo sets it. libfranka's torque interface takes torques *"without gravity and friction"*
(`control_types.h`), so gravity is compensated downstream and the controller correctly does not add
it — but that compensation is only as good as the payload model behind it.

An unmodelled PolyUMI end-effector therefore shows up as a **steady-state position offset under
load**, which is easy to misread as needing the integral term. It is not; `Ki` would be papering
over a wrong dynamics model. For scale, `touch_in_the_wild` configures 1.8 kg with a CoM of
(0.064, -0.06, 0.03) m for their comparable UMI gripper.

Sequenced last by choice, so the controller is characterised as-is first. **Move it ahead of step 7
if the deliberate-contact test behaves oddly** — a mis-modelled payload and a real contact force are
hard to tell apart, and this is the cheaper of the two to rule out.

### Choosing which executor a probe drives

Both executors take target poses, but in different message types on different topics, and aiming at
the wrong one is **silent** — the other executor simply never subscribes, so nothing moves and
nothing errors. `polyumi_ros2.target_chunk.Wire` is the single place that mapping lives:

| `wire` | Message | Topic | Consumer |
|---|---|---|---|
| `multidof` (default) | `MultiDOFJointTrajectory` | `/polyumi/target_poses_traj` | `polyumi_cartesian_impedance_controller`, **active** |
| `pose_array` | `PoseArray` | `/polyumi/target_poses` | `fr3_moveit_bridge` |

`tcp_pivot_test` and `latency_probe -p mode:=arm` both take `-p wire:=...` and default to the
streaming controller. `servo_smoke_test` only speaks the timed format — MoveIt cannot splice, so
the concept does not apply to it.

When a probe reports *"Nothing is subscribed to ... Needs: ..."*, read the consumer it names. For
the timed format that message says **ACTIVE, not merely loaded** — a controller spawned `--inactive`
holds no subscription, and that is indistinguishable from a crashed one from the publisher's side.

All of this goes away with step 9: once the MoveIt path is deleted, `Wire` collapses to one member
and the flags come out with it.
