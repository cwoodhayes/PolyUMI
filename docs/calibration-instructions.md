# System Calibration

PolyUMI depends on some hardcoded constants to run inference. Where possible, scripts are included in the repo to produce correctly calibrated values for these.
Some of these constants
should generally be the same from setup to setup (ie the gripper width offset), 
while others may be substantially different (sensor + arm execution latencies). 

Here is a quick doc detailing the calibration procedures for the system & where to find the associated values:

## Geometric Constants

### SLAM mask

**Change required when: any hardware revision that changes what the GoPro can see** — fingers, mirrors, LED strips, wiring, mount.

`ingest/config/slam_mask.png` blanks the camera-rigid hardware before ORB-SLAM3 tracks. Those pixels sit at a fixed image location no matter where the camera goes, so their features carry zero parallax (breaking two-view init) and give every keyframe the same DBoW2 signature (breaking relocalization). In practice this leads to widespread SLAM failures, which pretty much entirely go away after a correct mask (fully covering the device) is applied.

**White (non-zero) = discarded. Black = kept.** Inverting it is worse than no mask at all.

1. **Render a tracing template** from any mapping video. The temporal median leaves rigid hardware sharp and blurs the scene away — anything sharp is what you mask:
   ```bash
   uv run python -c "
   import cv2, numpy as np
   cap=cv2.VideoCapture('recordings/<scene>/<mapping_session>/gopro.mp4')
   n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); acc=[]
   for i in np.linspace(0,n-1,120).astype(int):
       cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,f=cap.read()
       if ok: acc.append(cv2.resize(f,(2704,2028)))
   cv2.imwrite('/tmp/trace_template.png', np.median(np.stack(acc),0).astype('uint8'))"
   ```

2. **Trace over the gripper** using the template in a photo editor. Paint the mask in solid white, and everything else black. Must use a 4:3 canvas (ie 2704x2028). Export RGB or black and white with no alpha. I used GIMP (Photoshop works too of course)--make a layer on top of the template image, then paint white over the gripper with the brush tool, and then hide the template image, and then export.

3. **Save to `ingest/config/slam_mask.png`** and check polarity + visually check fit:
   ```bash
   uv run pytest ingest/test/test_slam_step.py -q     # guards binary-ness and polarity
   uv run python -c "
   import cv2; m=cv2.imread('ingest/config/slam_mask.png',0)
   f=cv2.resize(cv2.imread('/tmp/trace_template.png'),(m.shape[1],m.shape[0])); f[m>0]=(0,0,255)
   cv2.imwrite('/tmp/mask_check.png',f)"                # red must cover every sharp edge
   ```

4. **Re-run SLAM on every affected scene** — `pingest pp 2 --scene <scene> --force`. The mask changes the map, so old atlases and old poses are not comparable to new ones.

#### Gotchas

- **Don't port UMI's polygons.** `umi/common/cv_util.py` describes *its* mount; ours has an extra PCB and mirrors further outboard, and its numbers misfit visibly. Upstream's `umi/asset/mask.json` is staler still — only `scripts/gen_image_mask.py` reads it, and it disagrees with the polygons the SLAM pipeline actually uses.
- **Path reaches the binaries via `Mask.Path` in the generated settings YAML**, alongside the atlas paths.

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
      execute_arm:=true execute_gripper:=true
ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"   # a roomy pose; edge poses fail to plan
ros2 run polyumi_ros2 tcp_pivot_test --ros-args -p angle_deg:=20.0
```

#### Gotchas

- **`franka_msgs/srv/SetTCPFrame` is not the right button to push here.** It changes `O_T_EE` reporting only. TF and MoveIt (currently used for motion planning) are driven entirely by the URDF.
- **Single source of truth for the transform is `nuc/tcp_calib.py`.** is the single definition; it reaches TF via a `static_transform_publisher` in `fr3_bringup.launch.py` and move_group's RobotModel via `nuc/description/fr3_polyumi.urdf.xacro`. Both read it from there.
- **Changing `T_gopro_to_fingertip` invalidates every dataset exported before the change**, since their poses are on the old body frame. Re-export and retrain; do not try to compensate at inference time.
- Residual error we deliberately don't chase: real GoPro mount tilt versus the CAD-nominal "optical axis parallel to the approach axis". Suspect it first if the policy is systematically off in *orientation* rather than position. Closing it would need a camera-based hand-eye calibration.

### End-effector payload

**Change required when: hardware design revision**

This is relevant for any setup using a Franka arm:

The FCI only cancels gravity for the mass it knows about (`m_ee` + `m_load`). Anything past the flange it *doesn't* know about — the GoPro, its mount, our fingers — is a constant force the cartesian impedance spring has to fight, so the TCP drops the instant that controller activates and holds a steady offset. Set in `nuc/tcp_calib.py` → `PAYLOAD_MASS`, `PAYLOAD_COM_HAND`; `fr3_bringup.launch.py` pushes it once per session by running `nuc/set_payload.py`, a one-shot `franka_msgs/srv/SetLoad` client.

First, find out what the robot already thinks it's carrying — this decides whether `PAYLOAD_MASS` is the whole assembly or just the part Desk isn't covering:

```bash
ros2 topic echo /franka_robot_state_broadcaster/robot_state --field inertia_ee   --once  # Desk's EE config; the Franka Hand is 0.73 kg
ros2 topic echo /franka_robot_state_broadcaster/robot_state --field inertia_load --once  # what SetLoad last applied
```

Then get a starting number. Weighing the unbolted assembly on a kitchen scale beats every indirect method (±5 g), and balancing it on a straight edge in two orientations gives the CoM to the ~1 cm that's plenty here. (fair warning I did not do this; eyeballing it as follows is probably good enough). Failing that, the arm will tell you, either from the droop you can already see or from its own force estimate:

```bash
# with the arm stationary and nothing touching it — this is the unmodelled payload's gravity wrench
ros2 topic echo /franka_robot_state_broadcaster/robot_state --field o_f_ext_hat_k.wrench   # m = -F_z / 9.81
```

To iterate without relaunching, `SetLoad` can be called by hand — but **only while no controller holds the arm.** With one active the robot is in `Move` mode and rejects it outright, so deactivate first and reactivate after. Note also that `center_of_mass` on the wire is in **`fr3_link8`** while `PAYLOAD_COM_HAND` is in `fr3_hand` — `tcp_calib.payload_com_flange()` does that conversion, and `tcp_calib.set_load_request()` prints the whole request literal for you. Write the winning numbers into `nuc/tcp_calib.py` when you're done.

What you really want to see, though, is that when you launch fr3_inference.launch.py,
(which starts the impedance controller upon first launch after the bringup), 
that the fingertip doesn't move at all in any direction, by rotation or translation, at _any position of the arm_.

You can kind of move the arm into different positions to back out if your CoM is at the wrong place once you have the
mass nailed down at the homing position (where you know that the 100% of the mass error goes into -z displacement).

Failing all else, you can sort of guess-and-check (and start with my existing numbers)
based on the CAD changes you've made, and watch for that result when you start up.


```bash
# NUC. Whichever of the two is active — `ros2 control list_controllers` tells you.
ros2 control switch_controllers --deactivate polyumi_cartesian_impedance_controller
ros2 service call /service_server/set_load franka_msgs/srv/SetLoad \
  "$(cd ~/Documents/PolyUMI/nuc && python3 -c 'import tcp_calib; print(tcp_calib.set_load_request())')"
ros2 control switch_controllers --activate polyumi_cartesian_impedance_controller
```

That sends what `fr3_bringup.launch.py` sends, so it is only useful once you have edited
`tcp_calib.py`. To try a number *before* committing to it, paste the request literal by hand in the
same shape — but read it out of `set_load_request()` first, so the CoM is in `fr3_link8` and the
inertia is non-zero. Both are easy to get wrong from scratch, and the FR3 hides the reason.

**Hardware verification:** converged when the TCP doesn't visibly move as the impedance controller activates and `o_f_ext_hat_k` sits near zero at rest. If it drops *upward*, the mass is now too high.

#### Gotchas

- **The URDF is not the lever.** `franka_hardware` reads only `robot_ip` and `arm_id` out of it, so an `<inertial>` block in `nuc/description/fr3_polyumi.urdf.xacro` changes nothing. Only `SetLoad` or Desk move `m_load`.
- **The droop is a measurement, not just a symptom**: $\Delta z = m_{unmodelled} \cdot g / K_{trans}$, so at `translational_stiffness: 2000` each mm of sag is 0.204 kg. But `translational_clip` is 0.01 m, so at ~1 cm the spring saturates at 20 N and the reading is pinned — the real mass is only *at least* 2 kg.
- `o_f_ext_hat_k` carries joint friction and arm model error too (~1–3 N, the same order as a 0.5 kg payload). Average it, and read it at two or three wrist orientations — the part that rotates with the tool is the payload.
- **Inertia may NOT be left at zero.** A nonzero mass with a zero tensor is physically impossible and the FR3 rejects it — the whole error you get back through ROS is `success: false, error: 'command exception error'`, with the real message (`Set Load command rejected: invalid argument!`) only in the `/service_server` log. `tcp_calib.payload_inertia_flange()` approximates it as a uniform solid box from `PAYLOAD_EXTENTS` and rotates it into the flange, where setLoad reads it — the same frame as `center_of_mass`, and not the `fr3_hand` frame the extents are stated in. It is a 3x3 matrix flattened to 9 elements, about the CoM. The box is assumed concentric with the CoM and uniformly dense, which `PAYLOAD_COM_HAND` already contradicts — accepted because inertia only enters the acceleration terms at speeds this low. Get the mass and the CoM right; the tensor only has to be plausible and non-zero. (UMI, SERL and polymetis all appear to leave inertia unset, but they reach the load through Desk, which derives a tensor for them.)
- **`SetLoad` only works when no controller is active.** Otherwise: `Set Load command rejected: command not possible in the current mode ("Move")`. This is why `fr3_bringup.launch.py` sequences the call ahead of the `fr3_arm_controller` spawner rather than alongside it.
- **A failed `SetLoad` aborts bringup**, rather than warning and carrying on. `nuc/set_payload.py` checks `response.success` and exits non-zero, and bringup refuses to spawn `fr3_arm_controller` unless it exited 0. (It is a client and not a `ros2 service call` precisely because the CLI exits 0 even when the body says `success: false` — gating on that would mean grepping its output for `success=True`, and a repr change would then read a *good* SetLoad as a failure.) The spawner is the point of no return — after it the robot is in `Move` mode and the payload cannot be set again without deactivating everything — and the only other symptom is TCP droop, which you have to be watching for. If bringup dies here, read the `/service_server` log before re-launching.
- **The service response tells you almost nothing.** `franka_param_service_server` catches every `franka::CommandException` and flattens it to the string `"command exception error"`. When a call fails, go read the `/service_server` log for the real reason: `grep -i "command exception" $(ls -t ~/.ros/log/ros2_control_node_*.log | head -1)`.
- If a residual sag survives a correct payload, that's joint friction, not payload. `translational_ki` in `nuc/config/polyumi_controllers.yaml` is the tool for it — SERL's approach — but reach for it second.

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

The policy is trained on time-synced observations (as aligned by the preprocessing pipeline `pingest pp`). On the robot, the observations are not naturally synchronized. We must calibrate or measure all the latencies in the system to be able to give the model the zero-latency, fully synchronized environment it was trained on.
We calibrate what we cannot measure in realtime, and place these calibration constants in `ros2_ws/src/polyumi_ros2/config/inference.yaml`.


```
photon ──(latency.gopro)──> header.stamp ──(measured live)──> response ──(latency.arm_exec)──> motion
        CALIBRATE                           nothing to do                CALIBRATE (loosely)
```

`header.stamp` is the earliest instant we can measure in our ROS client node orchestrating the inference process. Everything after that timestamp — the
YUYV convert, tick phasing, the POST, the network, the server's own inference time — is measured
empirically on every tick by `_n_stale_actions`, which runs *after* the response lands and computes
`now() - t_obs`, then turns it straight into a count of leading actions to discard.

| Value | Where it comes from |
|---|---|
| `latency.gopro` | **measure** — `latency_probe --ros-args -p mode:=camera` |
| `latency.arm_exec` | **measure** — `latency_probe --ros-args -p mode:=arm` |
| `latency.gripper_exec` | **measure** — `latency_probe --ros-args -p mode:=gripper_chirp`, not `mode:=gripper` — see below |
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
**refuses to print a line to paste** when the correlation peak is too weak or too broad to believe.

#### How to calibrate — `latency.gopro`

The method follows the original UMI: film a clock in the form of a QR code encoding the current time, then the
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

1. **Bring up the arm with execution on**, and get it somewhere roomy — edge poses fail to plan.
   `execute_arm:=true` activates the streaming impedance controller:
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
2. **Nothing between the fingers**, then run it. It needs no other measurement as input — the hand
   is truncated by its own latency alone, so this run is independent of `latency.arm_exec`:
   ```bash
   ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=gripper
   ```
   This one is a **step response**, not a chirp: it commands a 30 mm step, times how long until the
   fingers start moving, and repeats 8 times alternating direction. See the gotcha below for why
   cross-correlation is the wrong tool for this particular plant.
3. This prints **two** numbers, because they are two different quantities:
   - `latency.gripper_exec` — do **not** paste this one in. It is command → the hand actually
     moving, a few hundred ms, most of it the hand's own firmware (a `Move` blocks 363 ms even for
     zero travel). `franka_hand_node` already models that internally (`HandLimits.cmd_delay` in
     `gripper_trajectory_interpolator.hpp`) to decide which setpoint each `Move` can still reach —
     so feeding the same figure into `config/inference.yaml`'s `gripper_exec` would compensate for
     it twice. That field is currently `0.0` and should be left that way.
   - `latency.gripper` — the **observation** side, half the `/fr3_gripper/joint_states` publish
     interval. Goes in `config/inference.yaml`, unaffected by the above.


#### Gotchas

- **`latency.arm_exec` measures the arm's lag behind the equilibrium point**, under the streaming
  impedance controller. It is not a transport delay. `policy_client_node` anchors each chunk at
  `t_obs - arm_exec` so the equilibrium reaches pose k that far early and the arm, trailing it,
  arrives on time. The probe correlates against each waypoint's *intended* instant rather than its
  publication, so the probe's own `lead_s` cancels — re-run at a different `lead_s` and the answer
  should not move.
- **It now does two jobs, so it is worth more care than it used to be.** Besides feeding
  `_n_stale_actions`, it sets every chunk's anchor. Those two cancel in *timing* — the first
  published waypoint lands at ~now regardless — but not in *phase*: overstating it skips
  `arm_exec / action_dt` steps ahead in the policy's intended trajectory. Carrying the old MoveIt
  planning figure over to the servo would be several steps of skip.
- **The old ±20 ms bound was MoveIt's fault and is gone.** Correlation accuracy is set by excitation
  bandwidth, and the planner's cadence used to cap how fast the arm could be swept. The servo
  removes that cap, so expect a far sharper peak. A result still in the hundreds of milliseconds
  means something is routing through MoveIt — check `ros2 control list_controllers` before believing
  it. The probe chirps rather than using UMI's fixed sine for this reason; see the table in
  `polyumi_ros2/latency_util.py`.
- **A dead command path used to look like a marginal measurement.** If nothing is subscribed, or
  every waypoint is rejected as already elapsed, the arm never moves and the correlation is noise —
  which the sharpness check rejects with a message about peak width. The probe now detects this
  directly and names the likely causes instead.
- **The gripper is measured by step response, not cross-correlation, and that is deliberate.**
  `franka_hand_node` runs each `Move` to completion — a floor of 363 ms even for zero travel, and
  0.6–1.4 s for a real stroke — and drops anything inside its 5 mm deadband. Correlation
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
  (`robot_action_latency` vs `gripper_action_latency`), reached by slicing which leading actions
  get dropped rather than by shifting waypoint times. It means you can re-measure one device
  without touching the other, and a chunk too stale for the arm can still drive the hand. A
  `gripper_lead_steps` parameter on the old gripper bridge used to paper over the shared slice by
  indexing further into the chunk; it is gone, and re-adding a lead there would double-compensate.
  Both halves now carry an absolute schedule (`header.stamp + time_from_start`) numbered from the
  pre-slice index, so the drop removes waypoints without moving the ones that survive.
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