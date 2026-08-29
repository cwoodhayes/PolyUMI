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
| Pose body frame (training ↔ inference) | **done, verified on hardware** — `polyumi_tcp` end to end, CAD-measured, confirmed by `tcp_pivot_test` |
| Camera pixel transform (training ↔ inference) | **done** — shared crop+resize contract, pinned by cross-environment golden digests |
| Gripper command path | `franka_hand_node` written and unit-tested; **on-arm run pending** — tracked in [crb-fr3-inference.md](crb-fr3-inference.md), "Gripper problems", since the hand outlives this document |
| Receding-horizon inference stride | **done** — `steps_per_inference` (default 6) |
| DP export | **works**; UMI schema + tests landed. `export --type polyumi` adds the contact mic and the finger camera (`data/mic_0`, `data/finger_rgb`) |
| Real inference server | **in progress** — `serve_policy.py` green standalone on the GPU box; client wiring done + unit-tested; on-arm dry run pending hardware |

Everything else on the original list is done and has been deleted from here: DDS interop, the
round trip, gopro/proprio latency compensation, the `polyumi_tcp` body frame, the pixel transform,
gripper obs + width calibration, and the receding-horizon stride.

---

## What's left

1. **Finger cam + piezo are unwired on the inference side.** Params exist and are never consumed.
   The piezo is now exported — `pingest export --type polyumi` carries it as `data/mic_0`, see
   [maniwav-audio-policy.md](maniwav-audio-policy.md) — so the export half of this is done and the
   finger camera is what remains. What is left is the subscription itself, plus measuring
   `latency.finger_cam` and `latency.piezo_mic` with a rig. **This depends on the Pi being
   chrony-synced to the ROS host**; see "Clock sync" in [pi-provisioning.md](pi-provisioning.md),
   and note `./deploy.sh` warns when it is not. Note also that once either feeds the policy, the
   capture instant becomes the *oldest* across streams — an observation is only as fresh as its
   slowest signal.
