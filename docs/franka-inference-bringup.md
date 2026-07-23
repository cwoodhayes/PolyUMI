# Franka Inference Bringup Plan

Working document for incrementally bringing up diffusion_policy inference on the Franka arm.
Check off items as they are completed.

---

## Status

The **pose/vision path is structurally complete end-to-end** — latency compensation and frame
conventions both follow UMI, and single action chunks move the real arm. What's left is not
structural: it's two unmeasured constants, three unwired signals, and the training side.

| Workstream | State |
|---|---|
| FR3 NUC ↔ laptop over DDS | **done** — Phase 0 |
| Dummy server + client round trip | **done** — Phase 1 |
| Action-chunk execution on real hardware | **done** (single chunks); continuous 10 Hz loop unverified |
| Latency compensation, gopro + proprio | **done** — matches UMI's scheme; unit-tested |
| Latency compensation, finger cam + piezo | **not started** — params declared, never consumed |
| Pose body frame (training ↔ inference) | **half** — ingest emits `hand`; robot still reports `fr3_hand_tcp` |
| Gripper — observation | **not started** — hardcoded `0.0` |
| Gripper — command | **not started** — `action[7]` dropped before publish |
| DP export | **exists** for pose+image+gripper; no tactile; wrong schema for UMI; untested |
| Real inference server | **in progress** — Phase 3: `serve_policy.py` verified standalone on sheep; client dry-run wiring done + unit-tested (image 224, viz preview, `/reset`); on-arm dry-run pending hardware |

### Blocking issues

1. **`eef_frame` is the wrong frame.** Ingest step 5 exports poses on the canonical `hand`
   frame (anchored to the GoPro, the only body the handheld gripper and the FR3 share). The
   client still reads `fr3_hand_tcp`, which is the *Franka Hand* TCP — a different physical
   point. Until a static TF puts `eef_frame` on the `hand` frame, training and inference are in
   different frames and no policy can transfer. See [data-format.md](data-format.md).

2. **Two placeholder constants sit on critical paths**, both currently claiming more rigour
   than they have:
   - `latency.gopro: 0.05` in `config/latency.yaml`, commented "as measured by calibration
     scripts" — **no such script exists**. UMI measured 0.125–0.17 s for a plain UVC webcam;
     the GoPro→HDMI→capture-card→v4l2 path is longer, so 50 ms is likely well short. This now
     sets both the TF lookup instant *and* `t_obs` for chunk truncation.
   - `T_gopro_to_hand` in `ingest/config/gripper_calib.yaml` — translation inferred from
     `nominal_z_m`, rotation an unverified identity. Feeds every exported pose.

3. **Gripper is unwired in both directions.** `_lookup_agent_pos` hardcodes
   `gripper_width = 0.0`, so `agent_pos[7]` is a constant the policy learns nothing from; and
   `_actions_to_pose_array` drops `action[7]`, so a commanded width goes nowhere. A
   `PoseArray` cannot carry the width — the chunk topic needs a type that can, or a parallel one.

4. **The DP exporter needs a rework, and has no tests at all.** Deliberately deferred to one
   chunk of work rather than patched piecemeal, since the UMI migration rewrites this file
   anyway. Four things to fix together in `ingest/polyumi_ingest/export/dp/buffer.py`:
   - **It hard-requires OptiTrack even for SLAM-sourced poses.** `_export_episode` reads
     `optitrack/timestamps` unconditionally to clip the overlap window, so a SLAM-only scene
     raises `KeyError` *after* step 5 has written a perfectly good `eef/pose`. Predates the
     step-5 work (`--pose-source slam` failed the same way), but step 5 advertises a SLAM
     fallback the exporter silently contradicts. The window should clip to the sources the
     episode actually uses.
   - **Schema is Toby's, not UMI's.** A flat 8-vector `state`/`action`. `UmiDataset`
     name-matches its keys (`camera0_rgb`, `robot0_eef_pos`, `robot0_eef_rot_axis_angle`,
     `robot0_gripper_width`) and raises on anything it can't match; actions become 10-vectors
     (`pos(3) + rot6d(6) + gripper(1)`).
   - **No tactile.** Piezo audio and finger-camera frames aren't exported at all — the
     exporter touches `timestamps/finger` only to compute the window.
   - **The GoPro→finger clock shift is duplicated** between here (inline) and
     `eef_pose_step._gopro_ts_in_finger_clock`, and the two disagree on strictness: the
     exporter requires `annotations/time_sync`, the step defaults the offset to `0.0`. Not
     unified yet because picking one strictness is a behaviour change that belongs with this
     rework. (`nearest_idx` was the other duplicate; it now lives in `polyumi_ingest/timebase.py`.)

   No tests cover this file, which is how the OptiTrack coupling survived a contract change.
   Write them as part of the rework.

5. **Training stack not chosen in this doc yet.** The decision (recorded in conversation, not
   here) is to build on a fork of `universal_manipulation_interface` rather than base
   diffusion_policy, since UMI already has rot6d actions, relative pose frames, per-sensor
   latency in `shape_meta`, a `_target_`-swappable timm vision encoder, and — relevant to
   Phase 3 — `convert_pose_mat_rep(..., backward=True)`, which *is* the rel→abs conversion
   this doc specifies below. That changes the API contract's observation keys (see note there).

---

## Overall Architecture (target state)

```
GPU Machine                              Robot PC
─────────────────────────────────────    ──────────────────────────────────────
inference_server/ (uv, Python 3.12)        ROS2 (Kilted)
  ┌────────────────────────────────┐         ┌──────────────────────────────┐
  │  FastAPI server                │◄────────►│  policy_client_node.py       │
  │  POST /predict_cartesian/      │  HTTP    │  - buffers obs history       │
  │  - wraps DP inference          │  JSON    │  - POSTs to /predict_cartesian│
  │  - converts rel→abs actions    │          │  - executes returned EEF     │
  │  - returns abs EEF actions     │          │    targets via MoveIt2       │
  └────────────────────────────────┘          └──────────────────────────────┘
                                                           │
                                               MoveIt2 compute_cartesian_path
                                                           │
                                                    franka_ros2 / FCI
```

**Action space:** EEF Cartesian pose + gripper — `[x, y, z, qx, qy, qz, qw, gripper_width]` (8-vector).  
**Control frequency:** 10 Hz.  
**Inference location:** Phase 0–2 the (dummy) server runs on the laptop at
`localhost:8000`; Phase 3 moves it to a separate GPU machine called over LAN via HTTP.

> **Machine layout (FR3).** The "Robot PC" above is split: the **laptop** (Kilted)
> runs `policy_client_node`, MoveIt clients, camera, and Foxglove; the **FR3 NUC**
> (Humble) runs the Franka control stack and publishes `fr3_*` TF + joint states.
> They interoperate over CycloneDDS. See [Phase 0](#phase-0--fr3-nuc-bringup-distro--dds)
> and [crb-fr3-inference.md](crb-fr3-inference.md).

---

## API Contract

Observation key names currently match `shape_meta['obs']` in Toby's
`config/train_polyumi_image_diffusion_policy_cnn.yaml` so the server can pass them through
without remapping.

> **This contract changes when the training stack moves to UMI.** `UmiDataset` name-matches its
> keys (`sampler.py` counts robots via `key.endswith('eef_pos')`, picks Slerp vs linear interp
> via `'rot' in key`, and `get_normalizer` raises `RuntimeError('unsupported')` on any low-dim
> key it can't name-match), so a flat `agent_pos` 8-vector doesn't fit. The keys become
> `camera0_rgb`, `robot0_eef_pos`, `robot0_eef_rot_axis_angle`, `robot0_gripper_width`, and the
> action becomes a 10-vector `[pos(3), rot6d(6), gripper(1)]`. Treat the below as the *current*
> contract, not the target.

### `POST /predict_cartesian/`

**Request body:**
```json
{
  "n_obs_steps": 2,
  "n_action_steps": 8,
  "observations": {
    "image": {
      "dtype": "float32",
      "shape": [2, 256, 256, 3],
      "data":  "<base64 of the raw array bytes>"
    },
    "agent_pos": [[float×8, float×8]]
  }
}
```

- `n_obs_steps`: number of history frames being sent; must match the image's leading dimension.
- `n_action_steps`: how many action steps to return. Clamped server-side to the model's
  `n_action_steps` (currently **8** per training config); response echoes actual count.
- `observations` keys (matching `shape_meta`):
  - `image`: base64-encoded raw bytes plus `dtype`/`shape`, decoded server-side with
    `np.frombuffer(...).reshape(shape)`. Logically `[n_obs_steps, H, W, C]`, float32 in
    **[0, 1]**, RGB, H=W=**256**. Not nested JSON lists: at 256×256×3×`n_obs_steps` that's
    ~1.5 MB/frame of JSON and too slow to encode at 10 Hz.
  - `agent_pos`: `[n_obs_steps, 8]` — `[x, y, z, qx, qy, qz, qw, gripper_width]` in robot base
    frame (absolute). **`gripper_width` is currently always `0.0`** — see Status blocker 3.

**Coordinate convention (UMI):**
- **Observations** (`agent_pos`) are sent as **absolute** EEF coordinates in robot base frame.
  Note UMI itself feeds the policy *relative* obs — its dataset re-expresses obs relative to the
  latest obs step, which is what makes the policy translation-invariant. The conversion happens
  dataset-side, so what crosses this wire stays absolute.
- The DP model outputs actions as **relative** poses.
- The server converts relative → absolute before returning, using `agent_pos[-1]` from the
  request as the current EEF pose.

> **`agent_pos[-1]` is correct for UMI and wrong for Toby's dataset.** UMI's sampler is
> *now-anchored*: obs looks back from `current_idx`, actions look forward from it, so
> `action[0]` and `obs[-1]` are the same instant (`umi_dataset.py` uses `base_pose_mat=pose_mat[-1]`),
> and its policy returns the full horizon with no offset. Base DP is *window-anchored*: it slices
> `action_pred[:, To-1 : To-1+Ta]`, and Toby's `to_relative_action` takes `abs_action[0]` as
> origin — i.e. `agent_pos[0]`, the step DP discards. Inverting a Toby-trained model with
> `agent_pos[-1]` biases every chunk by one obs step (100 ms of motion at 10 Hz). Moving to UMI
> retires this rather than requiring a fix.

**Response body:**
```json
{
  "actions": [[float×8, ...]],
  "n_action_steps": int
}
```

- `actions`: list of `n_action_steps` targets, each `[x, y, z, qx, qy, qz, qw, gripper_width]`,
  in **absolute** robot base frame coordinates.
- `n_action_steps`: actual steps returned (≤ requested, ≤ model's 8).

**Error:** standard FastAPI 422/500 with `{"detail": "..."}`.

---

## Phase 0 — FR3 NUC bringup (distro + DDS)

Goal: get the Kilted laptop talking to the **FR3** NUC (Humble) over DDS so the
stream/inference demos run against the new arm. Full environment reference:
[crb-fr3-inference.md](crb-fr3-inference.md).

**Split topology:** PolyUMI's ROS2 nodes are distro-agnostic and run on the laptop
under Kilted; the Franka stack is Humble-only and stays on the NUC. They
interoperate at the DDS wire level — CycloneDDS, `ROS_DOMAIN_ID=0`, the `10.0.0.x`
link, and a matching **unicast** peer list (the NUC disables multicast). This phase
runs everything (including the dummy inference server) on the laptop; the move to a
separate GPU machine is Phase 3.

This replaces the earlier panda/"fer" assumptions:
- TF frames `panda_link0` / `panda_EE` → **`fr3_link0`** / **`fr3_hand_tcp`**
  (now `policy_client_node` params `base_frame` / `eef_frame`).
- `franka_fer_moveit_config` → the NUC's `franka_bringup` + `franka_fr3_moveit_config`
  (launched on the NUC, removed as a laptop rosdep).

- [ ] `sudo apt install ros-kilted-rmw-cyclonedds-cpp` on the laptop
- [ ] `ros2_ws/config/cyclonedds_laptop.xml` present (mirrors NUC peers/interface)
- [ ] `source setup_franka_env.sh` sets RMW/domain/URI and brings up `10.0.0.1/24`
- [ ] NUC `fr3-bringup` + `fr3-arm-controller` running
- [ ] laptop `ros2 node list` sees NUC nodes; `tf2_echo fr3_link0 fr3_hand_tcp` streams
- [ ] `rosdep install --rosdistro kilted` clean (no `franka_fer_moveit_config`)

---

## Phase 1 — Dummy server + policy client node

Goal: validate the full ROS2 ↔ server round-trip without a real checkpoint.

### 1.1 — `inference_server/` package

New `uv` package at repo root with `pyproject.toml`. Two server files:
- `dummy_server.py` — Phase 1, no torch, no ROS
- `server.py` — Phase 3, real inference (added later)

`dummy_server.py` behaviour:
- Sine-wave oscillator on X axis, ±0.05 m (`OSCILLATION_AMPLITUDE_M`) around a fixed home pose.
- Home pose set via env var `HOME_POSE`, parsed once at startup
  (`DEFAULT_HOME_POSE = '0.56 0.13 0.25 -1 0 0 0 0.4'` — xyz + quaternion + gripper width).
- Ignores `agent_pos`/image *content*, but validates their structure; 422 on missing/malformed fields.
- Returns `n_action_steps` copies of the oscillated pose (all identical, for simplicity).

**Run:**
```bash
cd inference_server
uv run dummy-server   # FastAPI on 0.0.0.0:8000 (the [project.scripts] entry point)
```

**Smoke test.** The image is base64'd raw bytes, so this shells out to python to build the body
rather than being pure curl:
```bash
python3 -c "
import base64, json, numpy as np
img = np.full((2, 256, 256, 3), 0.5, dtype=np.float32)
print(json.dumps({
    'n_obs_steps': 2, 'n_action_steps': 8,
    'observations': {
        'image': {'dtype': 'float32', 'shape': list(img.shape),
                  'data': base64.b64encode(img.tobytes()).decode()},
        'agent_pos': [[0.4, 0.0, 0.4, 0, 0, 0, 1, 0.04]] * 2,
    },
}))" | curl -s -X POST http://localhost:8000/predict_cartesian/ \
       -H "Content-Type: application/json" --data-binary @- | python3 -m json.tool
```

- [x] `inference_server/pyproject.toml` created (`fastapi`, `uvicorn`, `numpy` deps)
- [x] `dummy_server.py` implemented
- [x] smoke test returns `{"actions": [[...8 floats...]], "n_action_steps": 8}` with X oscillating across calls

---

### 1.2 — `policy_client_node`

**File:** `ros2_ws/src/polyumi_ros2/polyumi_ros2/policy_client_node.py`

**Subscribes:**
| Topic | Type | Purpose |
|---|---|---|
| `/gopro/image_raw` | `sensor_msgs/Image` | wrist camera via HDMI capture card → `v4l2_camera_node` (1920×1080@60, resized to 256×256) |
| TF `eef_frame` → `base_frame` | via `tf2_ros.Buffer` | absolute EEF pose (xyz + quat) |
| `/fr3_gripper/joint_states` | `sensor_msgs/JointState` | gripper width — **not implemented**, `agent_pos[7]` is hardcoded `0.0` |

**Timer:** `control_hz` (10 Hz).

**Logic per tick:**
1. Read the latest cached image **and its `header.stamp`**. Skip the tick if the frame is
   older than ~2 camera periods (the capture pipeline stalled).
2. Look up the EEF pose in TF at `image_stamp - latency.gopro + latency.proprio` — i.e. when
   *that frame* was captured, not now. tf2's buffer interpolates (linear + slerp) between
   cached transforms; `buffers.ee_pose_s` sizes its `cache_time` so the lookup stays in range.
3. Assemble `agent_pos = [x, y, z, qx, qy, qz, qw, gripper_width]`.
4. Append `(image, agent_pos)` to `deque(maxlen=n_obs_steps)`; if not yet full, skip (warn at 1 Hz).
5. Resize to `(image_height, image_width)`, normalize to [0, 1] float32, base64 the raw bytes.
6. POST to `/predict_cartesian/` with `n_obs_steps` and `n_action_steps` (8).
7. On success: drop leading actions already elapsed by execution time (`_n_stale_actions`
   measures `now() - t_obs` and adds `latency.arm_exec`), then log / publish the remainder.
8. On HTTP error / timeout: log and skip tick; do not raise.
9. The timer uses a `MutuallyExclusiveCallbackGroup`; if a previous tick's POST is still
   in flight when the next tick fires, that tick is skipped and a warning is logged.

**Latency model (mirrors UMI).** UMI corrects each sensor's *receive* timestamp back to true
capture time (`t_cal = t_recv - receive_latency`), then interpolates the low-dim streams onto
the camera's corrected clock. Steps 1–2 above are the same scheme, with tf2 playing the role of
UMI's `interp1d`/`Slerp`. Note UMI does **no** dataset-side latency shifting — every
`latency_steps` in its shipping configs multiplies by `dataset_frequeny: 0` and evaluates to
zero — because its training data is single-sensor (pose is SLAM *on the same GoPro frames*), so
there's no skew to correct. Ours has the same property, so `latency_steps: 0` is right for us
too, and **all** the compensation belongs here, on the robot. Cutting it would pair a fresh
pose with a stale image against a model trained on same-instant pairs.

**ROS2 parameters:**
| Name | Default | Description |
|---|---|---|
| `inference_server_url` | `http://localhost:8000/predict_cartesian/` | Server URL |
| `n_obs_steps` | `2` | History window (must match training config) |
| `n_action_steps` | `8` | Chunk size requested and published (≤ model's `n_action_steps`) |
| `image_topic` | `/gopro/image_raw` | Camera source |
| `control_hz` | `10.0` | Timer rate; also sets action spacing `action_dt` |
| `image_width` | `256` | Resize width (matches `shape_meta image: [3, 256, 256]`) |
| `image_height` | `256` | Resize height |
| `base_frame` | `fr3_link0` | TF base frame for the EEF lookup |
| `eef_frame` | `fr3_hand_tcp` | TF EEF/tool frame — **wrong frame**, see Status blocker 1 |
| `execute_motion` | `false` | Off by default: the arm does not move until explicitly enabled |
| `latency.gopro` | `0.0` | Camera capture→stamp delay. **Placeholder**, see Status blocker 2 |
| `latency.finger_cam` | `0.0` | Declared, **never consumed** |
| `latency.piezo_mic` | `0.0` | Declared, **never consumed** |
| `latency.proprio` | `0.0` | EEF-pose measurement delay |
| `latency.arm_exec` | `0.0` | Publish→arm-moves delay; used for chunk truncation |
| `buffers.ee_pose_s` | `1.0` | TF buffer `cache_time`; must exceed the largest compensated latency |

Defaults above are the node's own; the live values come from `config/latency.yaml` via
`inference_demo.launch.xml`.

**`package.xml` additions:** `tf2_ros`  
**`setup.py` addition:** `policy_client_node = polyumi_ros2.policy_client_node:main`

- [x] `policy_client_node.py` implemented
- [x] `package.xml` / `setup.py` updated
- [x] `colcon build` succeeds
- [x] node starts: `ros2 run polyumi_ros2 policy_client_node`
- [ ] with dummy server running: logs received 8-vector actions at 10 Hz (needs camera + TF — real hardware)

---

### 1.3 — Launch file

**File:** `ros2_ws/src/polyumi_ros2/launch/inference_demo.launch.xml`

Includes `stream_demo.launch.xml` (Foxglove bridge, `v4l2_camera_node` on the GoPro capture
card, and `pi_receiver_node` unless `motion_only`) and starts `policy_client_node` with
`config/latency.yaml` loaded via `<param from="..."/>`. See the file itself for the current
args — the ones you'll reach for:

```bash
ros2 launch polyumi_ros2 inference_demo.launch.xml
ros2 launch polyumi_ros2 inference_demo.launch.xml \
    inference_server_url:=http://<gpu-ip>:8000/predict_cartesian/   # Phase 3
ros2 launch polyumi_ros2 inference_demo.launch.xml \
    execute_motion:=true motion_only:=true   # FR3 motion only, no Pi needed
```

`execute_motion` defaults to **false** — the arm does not move until you ask it to.

- [x] `inference_demo.launch.xml` created
- [ ] launches cleanly against the dummy server (Phase 0: local `localhost:8000`):
  `ros2 launch polyumi_ros2 inference_demo.launch.xml`

---

## Phase 2 — MoveIt2 Cartesian execution — DONE, verified on hardware

**This section is historical planning; it predates implementation and got some things
wrong.** The as-built design, the reasons it differs from the plan below, and how to run
it live in [crb-fr3-inference.md](crb-fr3-inference.md#action-chunk-execution) — that's
the authoritative reference now, not this section.

What changed vs. the original plan, and why:

- **No `moveit_py`.** It can't run as a thin client to the NUC's move_group (it needs
  `robot_description`/SRDF loaded in-process, which requires the Humble-only
  `franka_description` — not available on the Kilted laptop). Superseded by raw
  `moveit_msgs` calls, matching a pattern already proven on this PC in a prior project.
- **MoveIt calls don't run in `policy_client_node` (laptop).** A `MoveGroup.Goal` /
  `GetCartesianPath.Request` sent laptop→NUC gets corrupted by the rmw-version gap
  (laptop rmw_cyclonedds 4.x vs NUC 1.x) — move_group logs `Catastrophic failure`. The
  MoveIt calls run in a separate node, `nuc/fr3_moveit_bridge.py`, **on the NUC**
  (same rmw as move_group, no corruption). `policy_client_node` publishes the target
  pose chunk as a `geometry_msgs/PoseArray` on `/polyumi/target_poses`; small messages
  cross the rmw gap fine.
- **Executes the whole action chunk, not just `actions[0]`.** A single-waypoint target at
  10 Hz is a discrete hop the arm can't track; the client requests `n_action_steps` (8)
  and publishes/executes it as one multi-waypoint Cartesian path (receding-horizon
  control).
- **Plan against SRDF group `fr3_arm`, not `fr3_manipulator`.** Only `fr3_arm` has an IK
  solver entry in `kinematics.yaml`; `fr3_manipulator` returns `fraction=0.0` for every
  Cartesian request on this Humble MoveIt version. `fr3_arm` still accepts
  `fr3_hand_tcp` as `link_name`.
- **No velocity-scaling field on Humble's `GetCartesianPath`** (added in a later MoveIt);
  the bridge time-scales the planned trajectory instead.
- **`franka_fr3_moveit_config/launch/move_group.launch.py` doesn't work as shipped** — it
  references launch args without declaring them, and omits the OMPL/controller/planning-
  scene-monitor params move_group needs to actually execute. `nuc/launch/fr3_move_group.launch.py`
  is a fixed, enriched copy.

- [x] `franka_ros2` / `franka_bringup` on the NUC (`fr3-bringup`)
- [x] MoveIt (`move_group`) running on the NUC via `nuc/launch/fr3_move_group.launch.py`
- [x] `fr3_hand_tcp` TF frame published and live on the laptop over DDS
- [x] `nuc/fr3_moveit_bridge.py` implemented: plans+executes chunks via local move_group
- [x] EEF target execution tested and verified moving the **real** robot (reduced velocity)
- [X] full 10 Hz dummy-sine loop run end-to-end (single chunks verified; continuous loop not yet)
- [ ] executing gripper control in addition to pose.

---

## Phase 3 — Real inference server

**Move to a dedicated GPU machine.** Through Phase 2 the (dummy) server runs on the
laptop at `localhost:8000`. Here it moves to a separate GPU box reached over LAN:
- Laptop gains a **second wired NIC** (USB-to-Ethernet) on its own subnet to the GPU
  machine — distinct from the `10.0.0.x` NUC link. Verify the adapter enumerates
  (`ip link` shows a second `enx*`) and that the two subnets / default route don't
  collide. The NUC ↔ laptop CycloneDDS link is unaffected (different interface).
- Point the client at it: `inference_demo.launch.xml inference_server_url:=http://<gpu-ip>:8000/predict_cartesian/`.
- DDS stays laptop↔NUC only; the GPU link is plain HTTP, so no Cyclone changes.

**Architecture decision — RESOLVED, and the original recommendation is superseded.** This doc
first recommended *subprocess* isolation (a `dp_worker.py` run via `conda run -n umi`, speaking
JSON over stdin/stdout) because `inference_server` was a uv/3.12 env and DP a separate conda/3.9
env. **The training work changed the premise:** it produced a single Docker image whose `umi`
conda env contains *both* `diffusion_policy`/torch *and* fastapi/uvicorn. So `serve_policy.py` (run
by `docker/serve.sh`) runs **inside that env and direct-imports the policy** — no subprocess, no
`dp_worker.py`, no `conda run`. The container already *is* the isolation boundary the subprocess
bought. `dummy_server.py` stays as the no-GPU mock for ROS-side dev/CI.

A DP/UMI checkpoint is **self-describing and dill-pickled** (`base_workspace.save_checkpoint`
writes `{'cfg', 'state_dicts', 'pickles'}`; `load_payload` un-pickles arbitrary objects), so the
serving env must match training — which the single image guarantees. Loading needs only a *path*
(~6 lines, cf. `base_workspace.load_payload`):

```python
payload   = torch.load(open(ckpt, 'rb'), pickle_module=dill, map_location='cpu')
cfg       = payload['cfg']                      # config travels IN the checkpoint
workspace = hydra.utils.get_class(cfg._target_)(cfg)
workspace.load_payload(payload)
policy    = workspace.ema_model                 # NOT .model — EMA is what eval uses
policy.to(device); policy.eval()
```

Then per request `policy.predict_action(obs_dict)['action_pred']`, where `obs_dict` is **batched,
channel-first torch tensors on device**. The wire↔UMI translation lives in `serve_obs.py`
(`wire_to_obs_dict`, `actions_rel_to_abs`), unit-tested without a checkpoint in
`test/test_serve_obs.py`.

Notes that will cost time if missed:
- **Use `ema_model`, not `model`.** The EMA weights are what eval uses; `.model` runs but is worse.
- **Normalization ships inside the checkpoint** (baked into the policy's state dict), so the server
  needs no stats — but a wrong action parameterization is likewise baked in permanently.
- **UMI's policy returns the full horizon with no offset** (`diffusion_unet_timm_policy` has no
  `n_action_steps`; read `result['action_pred'][0]`); truncation is the client's job.
- **Rel→abs is `convert_pose_mat_rep(..., backward=True)`** with `pose_rep='relative'`, i.e.
  `base_pose_mat @ pose_mat`. Don't re-derive it.
- **Image is 224×224, float32 [0,1]** (client already `/255`s); obs keys are the UMI names
  (`camera0_rgb`, `robot0_eef_pos`, `robot0_eef_rot_axis_angle`, `..._wrt_start`,
  `robot0_gripper_width`); the action is a 10-vec `[pos(3), rot6d(6), gripper(1)]`.

**Episode-start pose (`/reset`).** The policy consumes `robot0_eef_rot_axis_angle_wrt_start` —
orientation relative to where the *episode* began — but the wire `agent_pos` carries only the
*current* pose. `serve_policy.py` adds `POST /reset {agent_pos: [8]}` that caches the start pose;
`predict_cartesian` uses it, falling back to the current pose (with a warning) if `/reset` hasn't
run. **Follow-up:** once the client calls `/reset` at episode start, `dummy_server.py` needs a
matching **no-op `/reset`**, or dummy-based ROS bringup/CI breaks.

`serve_policy.py` over `dummy_server.py`:
- Startup (lifespan): validate `CKPT_PATH`, load `ema_model` into the process (direct import).
- `POST /reset` caches the episode-start pose; `GET /health` → `{status, checkpoint, device, ...}`.
- `POST /predict_cartesian/`: decode wire obs → `wire_to_obs_dict` → `predict_action` →
  `actions_rel_to_abs` → absolute EEF chunk.

- [x] subprocess vs. direct import → **single Docker image, direct import** (subprocess plan retired)
- [x] `serve_obs.py` translation helpers + `test/test_serve_obs.py` (no-checkpoint unit tests)
- [x] `serve_policy.py` body (load + `/reset` + `/predict_cartesian/`)
- [x] standalone smoke test with the 70-epoch checkpoint (health + synthetic predict) — green on sheep
- [x] `dummy_server.py` no-op `/reset` (contract parity)
- [x] client wiring: image→224, viz-only preview `PoseArray`, `/reset` at episode start,
      `post_timeout_s` param (unit-tested; `colcon test` clean)
- [ ] **arm dry-run** (`execute_motion:=false`): watch `/polyumi/target_poses_preview` in Foxglove
      — pending GoPro/arm hardware access
- [ ] end-to-end WITH execution (`execute_motion:=true`) — after the dry-run looks sane

### Arm dry-run procedure (no execution)

Validates the full pipeline and the *sanity* of commanded motion without moving the arm:
`execute_motion` stays `false`, so the node computes obs → POSTs → logs, and publishes the
commanded chunk **only** to the viz-only `/polyumi/target_poses_preview` (the NUC bridge never
subscribes to it). Accuracy is not the goal here — `eef_frame` is still `fr3_hand_tcp` (Q6), so
poses are structurally real but not spatially calibrated.

1. **Server on sheep**: `CKPT=/abs/path/to/epoch=0070-….ckpt ./serve_policy.sh` (builds the image,
   wires the checkpoint + HF cache; see [training-instructions.md](training-instructions.md)). From
   the laptop: `curl http://<sheep-ip>:8000/health` → `ready`.
2. **Laptop**: `source setup_franka_env.sh`; NUC publishing `fr3_*` TF; GoPro streaming.
   `ros2 launch polyumi_ros2 inference_demo.launch.xml inference_server_url:=http://<sheep-ip>:8000/predict_cartesian/`
3. **Watch**: node logs `mode: log-only (no motion)`, one `/reset` line, then per-tick chunk logs.
   Server `/health` shows `episode_start_set: true`. In **Foxglove** (`:8765`) add
   `/polyumi/target_poses_preview` — the chunk shows as pose arrows in `fr3_link0`, and should sit
   near the current EEF and step smoothly. Wild jumps / NaNs / off-workspace poses are the finding.

---

## Open questions

| # | Question | Status |
|---|---|---|
| 1 | Which package provides FCI control? | **Resolved:** NUC `franka_bringup` (`franka.launch.py arm_id:=fr3`) + `franka_fr3_moveit_config` controllers, run on the NUC. |
| 2 | Does DP receive `agent_pos` as absolute or relative to first obs frame? | **Resolved — and the original assumption was backwards.** UMI's convention is *relative*, not absolute: `umi_dataset.__getitem__` re-expresses obs relative to `pose_mat[-1]` (the latest obs), which is what makes the policy translation-invariant. Toby's `PolyUMIImageDataset` passes `agent_pos` through *absolute*, and fits one normalizer across all 8 dims incl. the quaternion. The transform is dataset-side either way, so this wire stays absolute — but "UMI convention = absolute" was wrong. |
| 3 | `moveit_py` availability — and on which machine (Phase 2)? | **Resolved:** not used at all. Raw `moveit_msgs` calls from a small node (`nuc/fr3_moveit_bridge.py`) running on the NUC, same-rmw as move_group — see Phase 2 / crb-fr3-inference.md. |
| 4 | Gripper width topic on Franka? | **Resolved:** `/fr3_gripper/joint_states`; actions `/fr3_gripper/{grasp,move,gripper_action,homing}`. |
| 5 | Subprocess vs direct import for Phase 3 | **Resolved — direct import (subprocess retired).** The original answer was subprocess, but the training work produced a single Docker image whose `umi` env has both `diffusion_policy`/torch and fastapi/uvicorn, so `serve_policy.py` direct-imports the policy and the container is the isolation boundary. See Phase 3. |
| 6 | Which physical point is the canonical `hand` frame, and what publishes it on the FR3? | **Open — blocking.** Ingest step 5 emits poses on a `hand` frame defined by `T_gopro_to_hand` (itself an unmeasured placeholder). The FR3 must report that same point as `eef_frame`; today it reports `fr3_hand_tcp`. Needs a mounting calibration + a static TF. |
| 7 | How do gripper width (obs) and gripper command (action) get wired? | **Open.** Obs: subscribe `/fr3_gripper/joint_states` and fill `agent_pos[7]` (currently `0.0`). Action: `PoseArray` can't carry width — either change the chunk message type or publish a parallel one, then drive `/fr3_gripper/{grasp,move}` from the NUC bridge. |
| 8 | Do finger cam / piezo feed the policy, and at what latency? | **Open.** Params exist and are unconsumed. If they become obs, the capture instant becomes the *oldest* across streams (an observation is only as fresh as its slowest signal), and they must also be added to the DP export, which carries none of them today. |
| 9 | What is `latency.gopro` actually? | **Open — needs measurement, not a guess.** See Status blocker 2. |
