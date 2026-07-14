# Franka Inference Bringup Plan

Working document for incrementally bringing up diffusion_policy inference on the Franka arm.
Check off items as they are completed.

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

Observation key names match `shape_meta['obs']` in `config/train_polyumi_image_diffusion_policy_cnn.yaml`
so the server can pass them through without remapping.

### `POST /predict_cartesian/`

**Request body:**
```json
{
  "n_obs_steps": 2,
  "n_action_steps": 1,
  "observations": {
    "image":     [[[[float, ...]]]], 
    "agent_pos": [[float×8, float×8]]
  }
}
```

- `n_obs_steps`: number of history frames being sent; must match array leading dimension.
- `n_action_steps`: how many action steps to return. Clamped server-side to the model's
  `n_action_steps` (currently **8** per training config); response echoes actual count.
- `observations` keys (matching `shape_meta`):
  - `image`: `[n_obs_steps, H, W, C]`, float32 in **[0, 1]**, RGB. H=W=**256** per training config.
  - `agent_pos`: `[n_obs_steps, 8]` — `[x, y, z, qx, qy, qz, qw, gripper_width]` in robot base frame (absolute).

**Coordinate convention (UMI):**
- **Observations** (`agent_pos`) are sent as **absolute** EEF coordinates in robot base frame.
- The DP model outputs actions as **relative** poses (first waypoint = origin, subsequent
  waypoints relative to it).
- The server converts relative → absolute before returning, using `agent_pos[-1]` from the
  request as the current EEF pose.

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
- Sine-wave oscillator on X axis, ±0.05 m around a configurable home pose.
- Home pose set via env var `HOME_POSE` (default: `"0.4 0.0 0.4 0 0 0 1 0.04"` —
  xyz + quaternion + gripper width).
- Oscillates around the fixed `HOME_POSE` (parsed once at startup); ignores `agent_pos`/image content.
- Validates required `observations` keys; returns 422 on missing fields.
- Returns `n_action_steps` copies of the oscillated pose (all identical, for simplicity).

**Run:**
```bash
cd inference_server
uv run dummy-server   # FastAPI on 0.0.0.0:8000 (the [project.scripts] entry point)
```

**Smoke test:**
```bash
curl -s -X POST http://localhost:8000/predict_cartesian/ \
  -H "Content-Type: application/json" \
  -d '{
    "n_obs_steps": 2, "n_action_steps": 1,
    "observations": {
      "image": [[[[0.5, 0.5, 0.5]]]],
      "agent_pos": [[0.4, 0.0, 0.4, 0, 0, 0, 1, 0.04],
                    [0.4, 0.0, 0.4, 0, 0, 0, 1, 0.04]]
    }
  }' | python3 -m json.tool
```

- [x] `inference_server/pyproject.toml` created (`fastapi`, `uvicorn`, `numpy` deps)
- [x] `dummy_server.py` implemented
- [x] smoke test returns `{"actions": [[...8 floats...]], "n_action_steps": 1}` with X oscillating across calls

---

### 1.2 — `policy_client_node`

**File:** `ros2_ws/src/polyumi_ros2/polyumi_ros2/policy_client_node.py`

**Subscribes:**
| Topic | Type | Purpose |
|---|---|---|
| `/gopro/image_raw` | `sensor_msgs/Image` | wrist camera (256×256 after resize) |
| TF `fr3_hand_tcp` → `fr3_link0` | via `tf2_ros.Buffer` (params `eef_frame`/`base_frame`) | absolute EEF pose (xyz + quat) |
| `/fr3_gripper/joint_states` (Phase 2) | `sensor_msgs/JointState` | gripper width (metres) |

**Timer:** 10 Hz.

**Logic per tick:**
1. Look up current EEF pose from TF; read latest image and gripper width from subscribers.
2. Assemble `agent_pos = [x, y, z, qx, qy, qz, qw, gripper_width]`.
3. Append `(image, agent_pos)` to `deque(maxlen=n_obs_steps)`.
4. If buffer not yet full, skip (warn at 1 Hz).
5. Resize image to `(image_height, image_width)`, normalize to [0, 1] float32.
6. POST to `/predict_cartesian/` with `n_obs_steps` and `n_action_steps=1`.
7. On success: log returned action (Phase 1) / execute it (Phase 2).
8. On HTTP error / timeout: log and skip tick; do not raise.
9. The timer uses a `MutuallyExclusiveCallbackGroup`; if a previous tick's POST is still
   in flight when the next tick fires, that tick is skipped and a warning is logged.

**ROS2 parameters:**
| Name | Default | Description |
|---|---|---|
| `inference_server_url` | `http://localhost:8000/predict_cartesian/` | Server URL |
| `n_obs_steps` | `2` | History window (must match training config) |
| `image_topic` | `/gopro/image_raw` | Camera source |
| `control_hz` | `10.0` | Timer rate |
| `image_width` | `256` | Resize width (matches `shape_meta image: [3, 256, 256]`) |
| `image_height` | `256` | Resize height |
| `base_frame` | `fr3_link0` | TF base frame for the EEF lookup |
| `eef_frame` | `fr3_hand_tcp` | TF EEF/tool frame for the EEF lookup |

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

```xml
<launch>
  <arg name="inference_server_url" default="http://localhost:8000/predict_cartesian/"/>
  <include file="$(find-pkg-share polyumi_ros2)/launch/stream_demo.launch.xml"/>
  <node pkg="polyumi_ros2" exec="policy_client_node" name="policy_client_node">
    <param name="inference_server_url" value="$(var inference_server_url)"/>
  </node>
</launch>
```

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
- [ ] full 10 Hz dummy-sine loop run end-to-end (single chunks verified; continuous loop not yet)

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

**Architecture decision:** subprocess isolation vs. direct import.

| | Subprocess (recommended) | Direct import |
|---|---|---|
| Python version | Server: 3.12 via uv; DP: 3.9 conda | Stuck with DP's 3.9 + conda |
| Interface | stdin/stdout or local ZMQ JSON between processes | Simple function call |
| Startup | Manages DP child process lifecycle | Single process |
| Deps | `inference_server` env stays minimal | Inherits all DP deps (torch, hydra, etc.) |

**Recommended approach:** subprocess, with a `dp_worker.py` that loads the checkpoint and
speaks newline-delimited JSON on stdin/stdout. The FastAPI server launches it at startup and
routes requests to it.

`server.py` additions over `dummy_server.py`:
- On startup: `subprocess.Popen(["conda", "run", "-n", "robodiff", "python", "dp_worker.py", ckpt_path])`
- `dp_worker.py`: loads checkpoint, reads JSON requests from stdin, writes JSON responses to stdout
- `GET /health` → `{"status": "ready", "checkpoint": "..."}`
- Server handles relative→absolute action conversion (DP outputs relative; client expects absolute)

- [ ] subprocess vs. direct import confirmed
- [ ] `dp_worker.py` implemented and tested standalone
- [ ] `server.py` wrapping `dp_worker.py` implemented
- [ ] smoke test with a real checkpoint
- [ ] end-to-end: `policy_client_node` → real server → real robot

---

## Open questions

| # | Question | Status |
|---|---|---|
| 1 | Which package provides FCI control? | **Resolved:** NUC `franka_bringup` (`franka.launch.py arm_id:=fr3`) + `franka_fr3_moveit_config` controllers, run on the NUC. |
| 2 | Does DP receive `agent_pos` as absolute or relative to first obs frame? | Assuming absolute (UMI convention) — confirm in dataset |
| 3 | `moveit_py` availability — and on which machine (Phase 2)? | **Resolved:** not used at all. Raw `moveit_msgs` calls from a small node (`nuc/fr3_moveit_bridge.py`) running on the NUC, same-rmw as move_group — see Phase 2 / crb-fr3-inference.md. |
| 4 | Gripper width topic on Franka? | **Resolved:** `/fr3_gripper/joint_states`; actions `/fr3_gripper/{grasp,move,gripper_action,homing}`. |
| 5 | Subprocess vs direct import for Phase 3 | TBD |
