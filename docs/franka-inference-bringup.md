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
**Inference location:** separate GPU machine, called over LAN via HTTP.

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
- Uses `agent_pos[-1]` from the request as the oscillation centre (ignores image content).
- Validates required `observations` keys; returns 422 on missing fields.
- Returns `n_action_steps` copies of the oscillated pose (all identical, for simplicity).

**Run:**
```bash
cd inference_server
uv run uvicorn dummy_server:app --host 0.0.0.0 --port 8000
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
| TF `panda_EE` → `panda_link0` | via `tf2_ros.Buffer` | absolute EEF pose (xyz + quat) |
| `/franka_gripper/joint_states` (TBD) | `sensor_msgs/JointState` | gripper width (metres) |

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

**ROS2 parameters:**
| Name | Default | Description |
|---|---|---|
| `inference_server_url` | `http://localhost:8000/predict_cartesian/` | Server URL |
| `n_obs_steps` | `2` | History window (must match training config) |
| `image_topic` | `/gopro/image_raw` | Camera source |
| `control_hz` | `10.0` | Timer rate |
| `image_width` | `256` | Resize width (matches `shape_meta image: [3, 256, 256]`) |
| `image_height` | `256` | Resize height |

**`package.xml` additions:** `geometry_msgs`, `tf2_ros`, `tf2_geometry_msgs`  
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

- [ ] `inference_demo.launch.xml` created
- [ ] launches cleanly against remote dummy server:
  `ros2 launch polyumi_ros2 inference_demo.launch.xml inference_server_url:=http://192.168.x.x:8000/predict_cartesian/`

---

## Phase 2 — MoveIt2 Cartesian execution

Goal: wire returned EEF targets into actual robot motion. Test in **demo/simulation mode first**.

### 2.1 — Prerequisites

- [ ] `franka_ros2` installed on robot PC
- [ ] `libfranka` version matches robot firmware
- [ ] `ros-kilted-moveit` installed
- [ ] `panda_EE` TF frame published: `ros2 run tf2_ros tf2_echo panda_link0 panda_EE`

### 2.2 — Cartesian execution in `policy_client_node`

Add `MoveGroupInterface` (via `moveit_py`) to the node. Per tick, after step 7 above:

```python
# pseudocode
def _execute_eef_target(self, action_8):
    target = PoseStamped()
    target.header.frame_id = 'panda_link0'
    target.pose = array_to_pose(action_8[:7])   # xyz + quat
    plan, fraction = move_group.compute_cartesian_path([target.pose], eef_step=0.01)
    if fraction > 0.9:
        move_group.execute(plan, wait=True)
    else:
        self.get_logger().warn(f'Cartesian plan only {fraction:.0%} complete, skipping')
    # gripper width: action_8[7] → send to gripper controller (TBD)
```

`wait=True` keeps it simple at 10 Hz; each execution should fit within 100 ms at moderate speeds.

- [ ] `moveit_py` importable in ROS2 node
- [ ] EEF target execution tested in demo mode
- [ ] back-and-forth motion from dummy server visible in RViz

### 2.3 — Real robot bringup

- [ ] FCI enabled on Desk UI
- [ ] `ros2 launch franka_fer_moveit_config moveit.launch.py robot_ip:=<IP>` starts cleanly
- [ ] joint states visible on `/franka_robot_state_broadcaster/...`
- [ ] `panda_EE` TF frame updating live
- [ ] dummy server back-and-forth runs on real robot (reduce amplitude first)

---

## Phase 3 — Real inference server

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
| 1 | `franka_ros2` vs `franka_fer_moveit_config`: which package provides FCI control? | TBD |
| 2 | Does DP receive `agent_pos` as absolute or relative to first obs frame? | Assuming absolute (UMI convention) — confirm in dataset |
| 3 | `moveit_py` availability in Kilted? | TBD |
| 4 | Gripper width topic on Franka: `/franka_gripper/joint_states`? | TBD |
| 5 | Subprocess vs direct import for Phase 3 | TBD |
