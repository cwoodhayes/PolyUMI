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

TODO:

- [ ] sensor latencies:
    - [ ] gopro -> ros inference client node
    - [ ] finger cam -> ros inference client node
    - [ ] piezo mic -> ros inference client node
    - [ ] EE cartesian pose -> ros inference client node
- [ ] ros -> inference server -> response (round trip delay of server call -- currently missing in inference.yaml)
- [ ] arm execution delay