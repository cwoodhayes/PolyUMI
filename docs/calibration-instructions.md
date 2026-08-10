# System Calibration

PolyUMI depends on some hardcoded constants to run inference. Where possible, scripts are included in the repo to produce correctly calibrated values for these.
Some of these constants
should generally be the same from setup to setup (ie the gripper width offset), 
while others may be substantially different (sensor + arm execution latencies). 

Here is a quick doc detailing the calibration procedures for the system & where to find the associated values:

## Geometric Constants
### Hand -> TCP frame

**Change required when: hardware design revision**

The TCP or "Tool Control Point" frame is the point whose pose we control with the model output. For PolyUMI (as in the original UMI) it is the point positioned in between the two gripper fingertips, level with the fingers' upper surface.

It is oriented in optical conventions (z forward, x right, y down).

The offset between this point and the built-in franka hand frame (or the hand frame of whatever arm you're using) is both arm & end-effector specific, and must be recalibrated for every update to PolyUMI's design + for every new platform it supports.

#### How to calibrate

There are 2 separate versions of this constant that must be set:

| Side | Expresses | Set it in |
|---|---|---|
| Training | GoPro optical frame -> fingertip midpoint | `ingest/config/gripper_calib.yaml` → `T_gopro_to_fingertip` |
| Inference | `fr3_hand` -> fingertip midpoint (`polyumi_tcp`) | `nuc/tcp_calib.py` → `TCP_XYZ`, `TCP_RPY` |

Both are essentially just measured in the CAD ([gripper](https://cad.onshape.com/documents/51445b7d15b8d189878323f1/w/358bf42f47b2b1f2a511decc/e/9a3e51ec7a29118eecf3283b?renderMode=0&uiState=6a791ac67f8f298cc031e0ea) and [EE](https://cad.onshape.com/documents/e674950e5409bace1adf9ce3/w/92b242e38e2c65427b8cb5db/e/0ded13219a9c097fb326bd02)), but with some fudge factors built in.
For the training constant, measure the center of the GoPro model's lens plate -> fingertip, then add in a bit of a fudge factor to offset the actual
camera sensor plane's position. The constant is expressed in gopro optical frame.
For the inference constant, this is split into a few values to help clarify how it was calculated. The only confusing part is locating the fr3_hand coordinate in the cad; I used the center of the gripper finger track on the Franka Hand CAD model, which I then verified using the pivot test (see below).

**Hardware verification (sanity checks):** 

For the training constants, you should be able to view the resulting frame in Foxglove for any sessions you recorded; check this visually.

For the inference constant, use the `tcp_pivot_test` as a helpful sanity check. It controls the arm by commanding pure rotations about each axis on the TCP. If you've set it correctly, you should be able to see that the actual fingertip point of the gripper stays still in the air while everything else rotates around it:

```bash
# NUC: bringup + inference, both execute flags on (this closes the hand itself), and SLOW
ros2 launch nuc/launch/fr3_inference.launch.py \
      execute_arm:=true execute_gripper:=true max_velocity_scaling:=0.05
ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"   # a roomy pose; edge poses fail to plan
ros2 run polyumi_ros2 tcp_pivot_test --ros-args -p angle_deg:=20.0
```

#### Gotchas

- **`franka_msgs/srv/SetTCPFrame` is not the right button to push here.** It changes `O_T_EE` reporting only. TF and MoveIt (currently used for motion planning) are driven entirely by the URDF.
- **Single source of truth for the transform is `nuc/tcp_calib.py`.** is the single definition; it reaches TF via a `static_transform_publisher` in `fr3_bringup.launch.py` and move_group's RobotModel via `nuc/description/fr3_polyumi.urdf.xacro`. Both read it from there.
- **Changing `T_gopro_to_fingertip` invalidates every dataset exported before the change**, since their poses are on the old body frame. Re-export and retrain; do not try to compensate at inference time.
- Residual error we deliberately don't chase: real GoPro mount tilt versus the CAD-nominal "optical axis parallel to the approach axis". Suspect it first if the policy is systematically off in *orientation* rather than position. Closing it would need a camera-based hand-eye calibration.

### Gripper Static Offsets

**Change required when: hardware design revision**

The PolyUMI gripper & the arm-mounted PolyUMI end-effector will read off finger positions differently, and also may have different ranges of motion in practice.

We assume that the preprocessing tool *pingest* (which measures the finger width using ArUco tags) and the arm (which likely measures with an encoder of some sort) both report metrically correct & consistent values with some static offset (i.e. a mm is a mm, less some constant). Said another way, we assume $q_ee = q_gripper + (ee_closed_mm - gripper_closed_mm)$.

In addition to the strictly necessary offsets above, we also measure & track the maximum opening for gripper and jaw, which isn't totally necessary at runtime but does help us understand what part (if any) of the policy's output range is unreachable by the arm's end effector.

| Term | What it is | Set in |
|---|---|---|
| **closed gripper width** | ArUco tag separation with the handheld gripper's fingers touching | `ingest/config/gripper_calib.yaml` → `closed_mm` |
| **open gripper width** | ArUco tag separation with the handheld gripper's fingers fully open | `ingest/config/gripper_calib.yaml` → `closed_mm` |
| **closed EE width** | the *arm's* EE jaw aperture with fingers touching | `ros2_ws/src/polyumi_ros2/config/inference.yaml` → `gripper_min_width_m` |
| **open EE width** | the arm's EE jaw aperture with fingers fully open | same file → `gripper_max_width_m` |

#### Part 1 — the handheld side (closed width)

1. **Record a calibration scene.** Open and close the gripper fully in front of the GoPro, several times, **holding it shut for a few seconds each cycle**. 

2. **Run preprocessing, then the calibration:**
   ```bash
   pingest pp --scene <scene>              # detect the finger tags
   pingest calibrate-gripper --scene <scene>
   ```

   Recommended: after processing, you should sanity check the scene + gripper values look OK in Foxglove.

3. **Sanity check the reported table.** It reports the raw min, a low-percentile ladder, and how many samples sit within 1 mm of the chosen value against how many a uniform sweep would have put there by chance. What you want to see is the low percentiles clustered tightly and the plateau tally comfortably beating chance. Two warnings can appear:
   - *"the gripper looks like it passed through the closed position rather than resting at it"* — the recording is unusable. Re-record with longer dwells.
   - *"the raw min is N mm below p1"* — informational. It means a stray PnP solve exists and was correctly excluded; UMI's plain `nanmin` would have taken it at face value.

4. **Paste the `closed_mm` / `open_mm` lines it prints** into `ingest/config/gripper_calib.yaml`.

> ⚠️ **`closed_mm` is load-bearing and changing it changes the meaning of every dataset exported afterwards.** The DP exporter subtracts it, so exported `robot0_gripper_width` is opening-from-closed rather than raw tag separation. Each buffer (the name for the Diffusion Policy training format) records the value it was built with as `meta.attrs['gripper_closed_width_m']`, so you can always check what a given buffer assumed.

#### Part 2 — the arm side (both apertures)

**Change required when: hardware design revision, or new embodiment**

This checks the arm EE jaw's range of motion. Ends up being sort of unnecessary for now because my finger design doesn't interfere with the range of motion, and the Franka Hand documents its range of motion correctly. But it was good to validate this assumption.
If bringing up a new arm, you should adapt & run this script there too to be safe.

1. **Bring up the arm with the gripper allowed to move.** Without `execute_gripper:=true` every command is a silent no-op and the probe measures nothing.
   ```bash
   ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true
   ```
   The bridge's clamp already defaults to the Franka Hand's own maximum (0.0817 m), so the fingers stop the open sweep first if anything does. Only pass `gripper_max_width` if you have deliberately lowered it.

2. **Clear the fingers and run the probe.** It drives both extremes several times and reports where the hand actually stopped.
   ```bash
   ros2 run polyumi_ros2 gripper_range_probe
   ```
   Do **not** run it while `policy_client_node` is up — they publish to the same topic.

3. **Check the spread before believing the mean.** `Move` applies no force and stalls on contact, so a closed endpoint that wanders more than ~1 mm between reps is not repeatable enough to calibrate against; the probe fails and tells you to make the endpoint force-defined instead (re-run with the bridge's `use_grasp_below_m` raised and a chosen `grasp_force_n`). The probe also asks the bridge for its clamp, so it can tell you whether the open endpoint was the fingers or the software stopping them.

4. **Paste the `gripper_min_width_m` / `gripper_max_width_m` lines** into `ros2_ws/src/polyumi_ros2/config/inference.yaml`.

#### Sanity check

You should look at the gripper in CAD & make sure the distance between the ARUCO tags is about the same as the one you measured. 

Currently, the handheld gripper opens ~6 mm wider than the arm can, so the top ~7% of the policy's commanded range saturates at `gripper_max_width_m`. That is expected, shows up as the policy's intent clipping rather than as an error, and is inherent to the two mechanisms having different strokes. A gap of *centimetres*, or the arm appearing to open wider than the handheld, means one of the two measurements is wrong.

## Latencies

**Change required when: new capture hardware, new arm, or a change to the motion control path.**

The policy is trained on observations that were all extracted from the same GoPro frames, so in the
dataset every stream shares one instant exactly. On the robot they do not: the camera frame you feed
the policy shows the world as it was ~150 ms ago, while TF will happily hand you the arm's pose from
*now*. Pairing those two is the error latency compensation exists to prevent, and
`policy_client_node` does it by looking every observation up at the camera's corrected capture
instant — UMI's scheme. The constants below are what "corrected" means, and they all live in
`ros2_ws/src/polyumi_ros2/config/inference.yaml`.

### Only one of them actually needs a rig

```
photon ──(latency.gopro)──> header.stamp ──(measured live)──> response ──(latency.arm_exec)──> motion
        CALIBRATE                           nothing to do                CALIBRATE (loosely)
```

`header.stamp` is the earliest instant the laptop has any handle on. Everything *after* it — the
YUYV convert, tick phasing, the POST, the network, the server's own inference time — is measured
empirically on every tick by `_n_stale_actions`, which runs *after* the response lands and computes
`now() - t_obs`, then turns it straight into a count of leading actions to discard.

**So there is deliberately no round-trip latency constant, and adding one would make things worse.**
A configured guess cannot beat a live measurement of the same span. If you want to see the number,
`policy_client_node` already logs it per tick (`latency_inference`). Upstream UMI ends up in the
same place from the other direction: it prints inference time and never subtracts it, absorbing it
through the `is_new` filter instead.

| Value | Where it comes from |
|---|---|
| `latency.gopro` | **measure** — `latency_probe -p mode:=camera` |
| `latency.arm_exec` | **measure** — `latency_probe -p mode:=arm` |
| `latency.gripper_exec` | **measure** — `latency_probe -p mode:=gripper` |
| `latency.gripper` | printed by the gripper run; half the joint-state publish interval |
| `latency.proprio` | adopted constant, ~0.001 — see below |
| round trip | nothing to do; measured live |
| `latency.finger_cam`, `latency.piezo_mic` | deferred, and blocked — see below |

Everything is measured by one node with three modes:

```bash
ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=<camera|arm|gripper>
```

Each run prints the number, two quality figures, and the exact line to paste, and drops the raw
series in an `.npz` so a marginal run can be re-judged without booking the hardware again. It
**refuses to print a line to paste** when the correlation peak is too weak or too broad to believe —
if you see `** DO NOT PASTE THIS **`, fix the run, don't squint at it.

#### How to calibrate — `latency.gopro`

This is the load-bearing one. It sets the TF lookup instant, the gripper-width lookup instant, and
`t_obs` for chunk truncation, so an error here mis-times *everything* the policy sees.

The method is UMI's: film a clock. The laptop displays a QR code encoding the current time, the
GoPro films the screen, and the lag is how far behind that encoded time the frame's ROS stamp runs.

1. **Bring up the camera only** — no arm needed, and nothing to install:
   ```bash
   ros2 launch polyumi_ros2 stream_demo.launch.xml motion_only:=true
   ```
2. **Point the GoPro at the laptop screen** so the screen fills the frame, in focus, without glare.
   Then run it. The window goes fullscreen; `q` stops early.
   ```bash
   ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=camera
   ```
3. **Sanity check before pasting.** UMI measures 0.125–0.17 s for this same
   GoPro → HDMI → Elgato chain. Far outside that band means the rig, not the pipeline — check you
   filmed the screen and not a reflection of it. A handful of decoded codes (`n=` in the output)
   means the QR was too small, blurred, or washed out.
4. **Paste `gopro:` into `config/inference.yaml`.**

The run also prints a **stamp → arrival** figure. That is a different quantity: `v4l2_camera` stamps
at dequeue and does its ~200 ms YUYV→RGB convert *afterwards*, so that cost sits after the stamp and
is not part of `latency.gopro`. It is what `max_image_age_s` has to tolerate — keep that parameter
above the max the probe reports, or good ticks get dropped as "capture pipeline stalled".

#### How to calibrate — `latency.arm_exec`

Chirps the commanded EEF pose sideways and cross-correlates it against where `polyumi_tcp` actually
went. **This moves the arm.**

1. **Bring up the arm with execution on**, and get it somewhere roomy — edge poses fail to plan:
   ```bash
   # NUC
   ros2 launch nuc/launch/fr3_inference.launch.py execute_arm:=true
   # laptop
   ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"
   ```
2. **Clear the workspace and run it.** Default sweep is ±3 cm on `y` for 20 s.
   ```bash
   ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=arm
   ```
3. **Paste `arm_exec:` into `config/inference.yaml`.**

#### How to calibrate — the gripper

1. **Bring up the gripper with execution on** — without `execute_gripper:=true` every command is a
   silent no-op and the probe measures nothing.
   ```bash
   ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true
   ```
2. **Nothing between the fingers**, then — passing your configured `latency.arm_exec`, which the
   step-count conversion needs (see below):
   ```bash
   ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=gripper
   ```
   This one is a **step response**, not a chirp: it commands a 30 mm step, times how long until the
   fingers start moving, and repeats 8 times alternating direction. See the gotcha below for why
   cross-correlation is the wrong tool for this particular plant.
3. This prints **two** numbers, because they are two different quantities:
   - `latency.gripper_exec` — the **action** side, command → the hand actually moving. Goes into
     `config/inference.yaml` exactly as measured; `policy_client_node` truncates the gripper chunk
     by this value alone. Expect a few hundred ms and a spread of roughly one
     `min_command_period_s`, most of which is the bridge's own command timer — that quantisation is
     real delay in service too, so it belongs in the number.
   - `latency.gripper` — the **observation** side, half the `/fr3_gripper/joint_states` publish
     interval. Goes in `config/inference.yaml`.

#### Hardware verification (sanity checks)

- **Trust the quality figures, not the number.** The arm run reports a *peak correlation* (is this a
  match at all?), a *peak width* (is the lag actually localised?), and whether the winner was
  *pinned* to the edge of the search window. A broad peak means the sweep was too slow or too small
  to pin the lag down — raise `chirp_f1_hz` or `amplitude_m`, or lengthen `duration_s`, and re-run.
  A pinned result is never a measurement, it is the search bound; see the gotcha below.
- **Check the lag is invariant to how much of the sweep you use.** A real transport delay does not
  care; if you re-analyse the saved `.npz` over the first half of the run and get a noticeably
  smaller number, what you measured is *phase lag* from a plant that cannot follow, not latency.
  This is the check that caught the gripper.
- **Confirm `arm_exec` is planning-bound.** Re-run the arm probe at two different
  `max_velocity_scaling` values. The number *should* move. If it does, you have confirmed it is a
  planning-and-execution distribution rather than a fixed transport delay, which is what it is today
  (see gotchas).
- **End to end.** With the measured values in, run inference dry (`execute_motion:=false`, watch
  `/polyumi/target_poses_preview` in Foxglove) and check the startup latency-budget line and the
  stale-action count look sane before executing.

#### Gotchas

- **`latency.arm_exec` is not a transport constant today.** `fr3_moveit_bridge` plans and executes
  each chunk synchronously and drops chunks that arrive mid-flight, so the number is a distribution
  dominated by MoveIt's planning time, and it shifts with `max_velocity_scaling`. Measure it at the
  scaling you actually run at. This is tolerable because it only feeds `_n_stale_actions`, in units
  of a 0.1 s `action_dt` — tens of milliseconds of error costs nothing there, and nowhere else. When
  the Phase 4 streaming controller lands (see
  [franka-inference-bringup.md](franka-inference-bringup.md)) it becomes UMI's
  `robot_action_latency`: smaller, sharper, and subtracted per waypoint instead.
- **The arm cannot be driven broadband, so its number carries roughly ±20 ms.** Correlation accuracy
  is set by excitation bandwidth, and MoveIt's cadence caps how fast the arm can be swept. This is
  why the probe chirps rather than using UMI's fixed sine, and why the arm is still the least
  precise of the three. See the table in `polyumi_ros2/latency_util.py`.
- **The gripper is measured by step response, not cross-correlation, and that is deliberate.**
  `fr3_gripper_bridge` quantises commands to `min_command_period_s` (0.25 s), supersedes each
  in-flight `Move` goal with the next, and drops anything inside its 5 mm deadband. Correlation
  assumes the response is a delayed *linear echo* of the command, which that is not. Driven at
  0.6 Hz on 2026-08-10 the hand fell most of a cycle behind and the estimator reported the phase
  lag — 1.2 s — as if it were a delay. The tell was that the answer grew with how much of the
  accelerating sweep it saw (0.41 s → 0.94 → 1.04 → 1.20); a transport delay is invariant to that.
  Don't "fix" this by slowing the chirp — the linear-echo assumption is still violated.
- **A cross-correlation result pinned to the search bound is the clamp, not an answer.** The same
  run returned exactly 1.000 s against the then-1.0 s bound. It slipped past the sharpness check
  because clipping the search window also clips the reported peak width, so a clamped result looks
  deceptively sharp (138 ms reported; 538 ms once unclamped). The probe now rejects pinned results
  outright and the bound defaults to 2.0 s, but if you widen it by hand, keep the check.
- **The arm and the hand are truncated independently, so neither number affects the other.**
  `_n_stale_actions` runs once per device, each with its own `latency.*_exec`, and the two chunks
  are published from separate slices of the same action list. This is UMI's split
  (`robot_action_latency` vs `gripper_action_latency`), reached by slicing rather than by absolute
  waypoint times, since a `PoseArray` carries no timing. It means you can re-measure one device
  without touching the other, and a chunk too stale for the arm can still drive the hand. A
  `gripper_lead_steps` parameter on `fr3_gripper_bridge` used to paper over the shared slice by
  indexing further into the chunk; it is gone, and re-adding a lead there would double-compensate.
- **`latency.proprio` is adopted, not measured, on purpose.** It means "true EE pose → the stamp on
  TF", and isolating it needs external ground truth of the true pose. libfranka stamps at read, so
  it is ~1 ms; UMI hit the identical wall and hardcodes `robot_obs_latency: 0.0001`. The `arm_exec`
  measurement returns `arm_exec + proprio`, and at 1 ms that is well inside its own noise floor.
  Building a rig to split them is not worth it.
- **`latency.finger_cam` and `latency.piezo_mic` are blocked, not just unmeasured.** Neither stream
  is subscribed by the inference path yet, and underneath there is a clock-domain bug waiting:
  `cam_streamer.py` sends picamera2's *monotonic* `SensorTimestamp` while `audio_streamer.py` sends
  *epoch* `time.time_ns()`, and `pi_receiver_node` republishes both as ROS stamps as if they shared
  a clock. Ingest repairs this offline via `FrameWallClock`; the live path does not. Fix that first —
  until then no latency number for those streams can mean anything.
- **Don't run any probe mode while `policy_client_node` is up.** The arm and gripper modes publish to
  the same topics the policy does, and the bridges act on whichever chunk arrived last.
- **Monitor scanout is inside `latency.gopro`.** The probe subtracts its own render time but cannot
  see the panel's. That is a ~5–20 ms floor, and UMI does not correct for it either. A photodiode
  would remove it; it is not worth the rig.
- **Don't swap the QR decoder back to `detectAndDecodeCurved`.** UMI's camera script uses it, but on
  OpenCV 4.6 it decodes nothing off a flat screen — not even a synthetic code — so the probe uses
  `detectAndDecode` instead. Ported verbatim it would have read zero codes and looked like bad aim.
  `test_latency_probe.py` pins this.
- **The training side needs no latency configuration.** `polyumi.yaml` keeps `dataset_frequeny: 0`,
  which zeroes every `latency_steps` expression, and that is correct: image, SLAM pose, and ArUco
  width all come from the *same* GoPro frames, so their relative latency in the dataset is
  structurally zero. UMI ships the same setting for the same reason. All latency compensation lives
  on the robot.