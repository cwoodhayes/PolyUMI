# Franka Inference Bringup — plan & progress

**Scope: planning and progress tracking, not reference documentation.** When something here is
finished, the truth moves into the code and its docstrings, and the entry here gets deleted rather
than annotated. This whole document gets deleted once the pipeline is confirmed working. Reach for
these instead:

| For | Read |
|---|---|
| How to run the stack, and what breaks | [crb-fr3-inference.md](crb-fr3-inference.md) |
| Pose/gripper/image data conventions | [data-format.md](data-format.md) |
| Measuring the constants | [calibration-instructions.md](calibration-instructions.md) |
| The on-arm executor, and why torque control | `nuc/polyumi_fr3_controllers/include/.../cartesian_impedance_controller.hpp` |
| Gains, clips, collision thresholds | [polyumi_controllers.yaml](../nuc/config/polyumi_controllers.yaml) |
| Serving a checkpoint | `external/polyumi_diffusion_policy/serve_policy.py` |

---

## Status

The pose/vision path is structurally complete end to end, and the streaming impedance controller
holds the arm through a synthetic trajectory. What remains is the policy end-to-end on the arm,
the unwired tactile signals, and the training side. The Franka Hand is deprecated — it is being
replaced — so its remaining work is tracked with the hand itself, in
[crb-fr3-inference.md](crb-fr3-inference.md), not here.

| Workstream | State |
|---|---|
| Action-chunk execution on hardware | Single chunks and a synthetic overlapping-chunk trajectory pass. **Policy-driven, continuous 10 Hz loop unverified** |
| Latency compensation, finger cam + piezo | **not started** — params declared, never consumed |
| Gripper command path | `franka_hand_node` written and unit-tested; **on-arm run pending** — tracked in [crb-fr3-inference.md](crb-fr3-inference.md), "Gripper problems", since the hand outlives this document |
| DP export | **works**; UMI schema + tests landed. Still no tactile, and the rework below is outstanding |
| Real inference server | **in progress** — `serve_policy.py` green standalone on sheep; client wiring done + unit-tested; on-arm dry run pending hardware |

Everything else on the original list is done and has been deleted from here: DDS interop, the
round trip, gopro/proprio latency compensation, the `polyumi_tcp` body frame, the pixel transform,
gripper obs + width calibration, and the receding-horizon stride.

---

## What's left

1. **Finger cam + piezo are unwired.** Params exist and are never consumed. If they become
   observations, the capture instant becomes the *oldest* across streams — an observation is only
   as fresh as its slowest signal — and they must be added to the DP export, which carries neither.
   Open question inside this one: do they feed the policy at all, and at what latency?

2. **DP exporter rework.** Deferred as one chunk rather than patched piecemeal:
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

3. **The on-arm sequence**, below.

---

## The on-arm sequence

**This is the live worklist.** Steps 1–3 (activate, push by hand, synthetic overlapping-chunk
trajectory) and 8 (`latency.arm_exec`, 0.0810 s) passed on hardware 2026-08-19 and have been
deleted; what they established now lives in the controller and in
[polyumi_controllers.yaml](../nuc/config/polyumi_controllers.yaml).

Steps are ordered so each leaves exactly one new thing unproven. This is a **torque controller**:
no firmware guarantees stability, so a step that looks boring is a step doing its job. Do not skip
ahead to the policy.

`fr3_bringup` is required for every step — it owns the `controller_manager` and publishes the
`polyumi_tcp` static TF the controller looks up on activation. Steps 4 onward also need
`fr3_inference execute_arm:=true`. Enable FCI in the Desk UI first.

- [ ] **4. `/polyumi/home` round trip.** `ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"`.
      **Pass:** it switches out, homes, switches back, and the arm holds still afterwards.
      `ros2 control list_controllers` should show the impedance controller `active` again.

- [ ] **5. `tcp_pivot_test` through the impedance controller.** Defaults to the timed wire format;
      `-p wire:=pose_array` still drives the MoveIt path.
      **Pass:** closed fingertips stay visibly still. Same check that validated the TCP under
      MoveIt; here it additionally validates the Jacobian shift, which nothing else does in the
      room.

- [ ] **6. Arm dry run with the policy, no execution** (`execute_motion:=false`, the default).
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

- [ ] **7. End-to-end with the policy, executing.** The finding to watch for is not accuracy, it is
      **continuity across chunk boundaries** — no stop-and-go at 10 Hz. Compare the motion against
      `/polyumi/target_poses_preview` in Foxglove. If the arm holds still while everything else
      looks healthy, see [crb-fr3-inference.md](crb-fr3-inference.md), "When it doesn't come up".

- [ ] **8. Deliberate contact.** Drive into a fixed surface.
      **Pass:** the arm complies and does not trip Franka's collision monitor. If it does, raise
      `collision.*` in [polyumi_controllers.yaml](../nuc/config/polyumi_controllers.yaml) — the
      controller applies those itself and refuses to configure if they do not take, so a fault
      here means the values are still too low for this task, not that nothing is applying them.
      **Do step 9 first if this behaves oddly** — a mis-modelled payload and a real contact force
      are hard to tell apart, and the payload is the cheaper of the two to rule out.

- [ ] **9. Configure the end-effector load.** Nothing in this repo sets it, and the firmware's
      gravity compensation is only as good as the payload model behind it. An unmodelled
      end-effector shows up as a **steady-state position offset under load**, which is easy to
      misread as needing the integral term — it is not. Do it programmatically
      (`franka_msgs/srv/SetLoad`) so the value lives in the repo next to the gains. For scale,
      `touch_in_the_wild` configures 1.8 kg at a CoM of (0.064, -0.06, 0.03) m for a comparable
      UMI gripper.

- [ ] **10. Retire the MoveIt executor from the inference path.** Only after 7 passes. Delete
      `_on_target`, `_plan_cartesian`, `_run_execute` and the `PoseArray` subscription from
      `nuc/fr3_moveit_bridge.py`, and the `PoseArray` command publisher (not the preview) from
      `policy_client_node.py`. The node stays for `/polyumi/home`. `target_chunk.Wire` then
      collapses to one member and the `wire`/`executor` flags come out with it.
