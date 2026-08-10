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
conventions follow UMI, and single action chunks move the real arm. What remains is two unmeasured
latencies, three unwired signals, the executor redesign, and the training side.

| Workstream | State |
|---|---|
| FR3 NUC ↔ laptop over DDS | **done** |
| Dummy server + client round trip | **done** |
| Action-chunk execution on hardware | **done** for single chunks; continuous 10 Hz loop unverified. Executor redesign is Phase 4 |
| Latency compensation, gopro + proprio | **done**, matches UMI's scheme, unit-tested. The *values* are still guesses |
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

1. **`latency.gopro` is a guess.** Currently `0.13` in `config/inference.yaml`, marked
   "TODO measure". UMI measured 0.125–0.17 s for a plain UVC webcam and uses **0.17** for the same
   GoPro→HDMI→capture-card path we have (`uvc_camera.py:241`, `eval_real.py:186`), so ours is in
   the right neighbourhood but unverified. It sets both the TF lookup instant and `t_obs` for
   chunk truncation, so it is not cosmetic. `proprio` (0.03) and `arm_exec` (0.03) are guesses too.

2. **`latency.gripper` unmeasured.** UMI keeps it separate from the arm's
   (`gripper_action_latency` vs `robot_action_latency`).

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
- **Small flat messages cross fine.** `PoseArray` and `trajectory_msgs/JointTrajectory` were both
  verified byte-for-byte across the gap. Service calls also work — `gripper_range_probe` queries
  the NUC bridge's parameter service from the laptop. `MultiDOFJointTrajectory` is nested deeper
  and still warrants its own check before Phase 4 Stage 1 relies on it.
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

- **`franka_msgs/srv/SetTCPFrame` is not the lever** for the TCP. It only changes libfranka's
  `O_T_EE` reporting; TF and MoveIt are driven entirely by the URDF.
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
- **`agent_pos[-1]` is the right anchor for UMI and wrong for Toby's dataset.** UMI's sampler is
  *now-anchored*: obs looks back from `current_idx`, actions look forward, so `action[0]` and
  `obs[-1]` are the same instant (`umi_dataset.py` uses `base_pose_mat=pose_mat[-1]`). Base DP is
  *window-anchored* and Toby's `to_relative_action` takes `abs_action[0]`. Inverting a Toby-trained
  model with `agent_pos[-1]` biases every chunk by one obs step. Only matters if an old checkpoint
  resurfaces.

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

## Phase 4 — streaming Cartesian servo (designed, not started)

**Plan of record for closing the inference-latency / jerky-motion gap.** Written 2026-07-22 after
tracing upstream UMI (`../universal_manipulation_interface`) and live-probing the NUC.

### Why

Our on-arm path treats camera + inference latency as a *failure* (drop the tick / drop the chunk)
and drives the arm **plan-then-execute**. UMI treats that latency as an expected, compensated
quantity and drives the arm with a **continuous 1 kHz interpolated Cartesian servo**. The second is
why UMI is smooth and latency-tolerant. This is architectural, not a tuning constant:
plan-then-execute starts each chunk from rest, stops at its end, and discards the policy's intended
`dt` timeline.

UMI's executor (`franka_interpolation_controller.py:277-355`) is a 1 kHz loop around a
`PoseTrajectoryInterpolator` (`umi/common/pose_trajectory_interpolator.py`, pure numpy). Every 1 ms
it evaluates the interpolator and writes the desired EE pose. New chunks arrive as
`schedule_waypoint(pose, target_time)`, splicing each future waypoint in at its absolute time with
no stop. Action chunks are timestamped in absolute wall-clock anchored to the observation
(`action_timestamps = arange(len)*dt + obs_timestamps[-1]`, `eval_real.py:503`); in-past waypoints
are dropped.

### Remaining gaps

| # | Gap | UMI | Ours | Fix |
|---|-----|-----|------|-----|
| 1 | **Executor model** | 1 kHz interpolated servo, splices chunks | `compute_cartesian_path` + blocking `ExecuteTrajectory`, skip-while-busy | Streaming Cartesian-pose controller around a ported `PoseTrajectoryInterpolator` |
| 2 | **Action timing** | absolute wall-clock per waypoint | `PoseArray`, no per-waypoint time; NUC re-times the chunk | Carry per-waypoint absolute times |

Inference cadence was the third gap and is **closed** — `steps_per_inference` (default 6) already
strides the laptop side.

Gap 2 needs both machines to agree on wall-clock time, which is **now true** after the chrony sync
(CLAUDE.md, "Clock sync"). Absolute action timestamps from the laptop are meaningful on the NUC.

### NUC capability findings (live probe 2026-07-22, fr3-bringup up)

Determines the UMI-faithful executor is achievable natively. Expensive to re-derive, so kept:

- **`controller_manager` runs at 1000 Hz** (`franka_bringup/config/controllers.yaml`).
- **`franka_hardware` exposes a native Cartesian-pose command interface**: `0/cartesian_pose …
  15/cartesian_pose` — 16 doubles, column-major 4×4 `O_T_EE`, driven by libfranka's Cartesian pose
  motion generator. The direct ros2_control analogue of UMI's `update_desired_ee_pose`. Seed state
  interface `0..15/initial_cartesian_pose` gives the pose at activation.
- Also present: `vx..wz/cartesian_velocity`, per-joint position/velocity/effort, elbow interfaces;
  state `fr3/robot_model`, `fr3/robot_state`.
- **`moveit_servo` is NOT installed** — the ROS-native servo shortcut is out unless added.
- **`franka_example_controllers` is NOT built** on this NUC even though `controllers.yaml`
  references it. Its `CartesianPoseExampleController` is the natural template, so it must be built
  first (or the pieces vendored).
- `joint_trajectory_controller` is available — the fallback, but joint-space (needs IK) and its
  trajectory-replacement semantics are less smooth. Not recommended.

### Staged design

**Stage 1 — laptop, low-risk.** Publish a **timestamped trajectory** instead of a bare `PoseArray`:
`trajectory_msgs/MultiDOFJointTrajectory`, per-point `transform` + `time_from_start`, with
`header.stamp = t_obs`. Carries UMI's `(poses, action_timestamps)` exactly. Keep the client-side
coarse stale-drop; the NUC does fine scheduling.

**Stage 2 — NUC, the real work.** Port `PoseTrajectoryInterpolator` (pure numpy, small). Write a
ros2_control controller (C++, templated on `CartesianPoseExampleController`) that claims
`<i>/cartesian_pose`, seeds the interpolator from `initial_cartesian_pose` on activation,
subscribes to the Stage-1 topic and `schedule_waypoint`s each future pose (wall→monotonic
translated), and in `update()` at 1 kHz writes `pose_interp(now)` → `O_T_EE`. This retires
`fr3_moveit_bridge` from the inference path; keep it for homing/point-to-point.

**Stage 3 — latency model + tune.** Adopt `camera_obs_latency ≈ 0.17` as `latency.gopro`; tune
stiffness and `steps_per_inference`.

### Risks

- **libfranka continuity limits (the #1 hazard).** The Cartesian-pose motion generator rejects
  discontinuous commands (velocity/accel/jerk) → `cartesian_reflex` / communication errors. The
  interpolator is mandatory — never step the target — and activation MUST seed from
  `initial_cartesian_pose` so the first command matches the current pose.
- **Stiff position, not impedance.** `cartesian_pose` is position-controlled, unlike the compliant
  Cartesian impedance UMI drove via polymetis. Probably fine and arguably more accurate; if
  compliance is needed, a custom Cartesian-impedance effort controller is the fallback.
- **The gripper cannot join this redesign** — see the hardware ceiling above. It keeps its own
  channel and its own discrete commander.

### Checklist

- [ ] Stage 1: publish `MultiDOFJointTrajectory` (header.stamp `t_obs`, per-point `time_from_start`)
- [ ] Stage 1: verify it crosses laptop↔NUC DDS intact
- [ ] Stage 2: port `PoseTrajectoryInterpolator`
- [ ] Stage 2: build `franka_example_controllers` on the NUC as the template
- [ ] Stage 2: write the streaming controller (claim `cartesian_pose`, seed from
      `initial_cartesian_pose`, subscribe + `schedule_waypoint`, 1 kHz `update()`)
- [ ] Stage 2: dry-run with a synthetic slow trajectory, no policy — smooth, no reflex
- [ ] Stage 2: end-to-end with the policy; retire `fr3_moveit_bridge` from the inference path
- [ ] Stage 3: measure `latency.gopro` (~0.17 target), tune stiffness + stride

---

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
- [ ] **End-to-end with arm execution** (`execute_motion:=true`) — only after the dry run looks sane
- [ ] **Measure `latency.gripper`**
- [ ] **Continuous 10 Hz loop** end-to-end (single chunks verified; the loop is not)

---

## Still-open questions

| # | Question | Status |
|---|---|---|
| 1 | Do finger cam / piezo feed the policy, and at what latency? | **Open.** Params exist, unconsumed. See "What's left" 3. |
| 2 | What is `latency.gopro` actually? | **Open — needs measurement, not a guess.** UMI's 0.17 for the same capture path is the starting estimate. |
| 3 | Does `MultiDOFJointTrajectory` cross the rmw gap intact? | **Open.** `PoseArray` and `JointTrajectory` do; this one is nested deeper. Blocks Phase 4 Stage 1. |
