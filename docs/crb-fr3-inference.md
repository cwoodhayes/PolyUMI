# CRB FR3 Inference Setup

**Note for Northwestern CRB members**: This document describes how to run inference on the CRB lab's Franka FR3 arm (in the student office, connected to the NUC with the skull on it). 

**Note for users outside of Northwestern**: This is specific to our equipment, but is likely still useful as an example to bring up inference on your own equipment. This documents the specific two-machine
setup used in the CRB lab to drive a **Franka FR3** for PolyUMI inference. The
distro split, IP plan, DDS choice, and NUC aliases below are particular to this
hardware — adapt them for your own robot and network rather than copying verbatim.
For the lab-agnostic inference architecture and API contract, see
[franka-inference-bringup.md](franka-inference-bringup.md).

It captures the laptop and NUC environments and the DDS contract that lets a Kilted
laptop talk to a Humble NUC. If something here drifts from reality, fix it here —
`setup_franka_env.sh` and `ros2_ws/config/cyclonedds_laptop.xml` assume these values.

## Setup
### Topology

```
Laptop (Kilted, Noble)                        NUC (Humble, Jammy)  [nu-crb]
RMW=rmw_cyclonedds_cpp, DOMAIN=0  ◄─ DDS over ─►  RMW=rmw_cyclonedds_cpp, DOMAIN=0
enp0s31f6 = 10.0.0.1/24            10.0.0.x      enx00249b860356 = 10.0.0.2/24
  - foxglove_bridge                               - fr3-bringup (franka_bringup, arm_id:=fr3)
  - v4l2_camera (GoPro)                           - fr3-arm-controller
  - pi_receiver_node                              - franka_fr3_moveit_config / move_group
  - policy_client_node ──HTTP──┐                  - publishes fr3_* TF + joint states
  - dummy_server (localhost:8000) ◄┘              - enp89s0 = 192.168.51.10 → robot @ .20
```

The PolyUMI ROS2 nodes use only distro-agnostic APIs (`rclpy`, `sensor_msgs`,
`tf2_ros`, `foxglove_msgs`), so they run on the laptop under Kilted. The Franka
stack is Humble-only and stays on the NUC; the two machines interoperate purely at
the DDS wire level.

### User PC (i.e. my personal Ubuntu laptop)

| | |
|---|---|
| OS | Ubuntu 24.04 Noble |
| ROS2 | Kilted |
| Wired NIC | `enp0s31f6`, static **`10.0.0.1/24`** via NM profile `fr3-link`, direct cable to the NUC's `enx` |
| RMW | `rmw_cyclonedds_cpp` — `sudo apt install ros-kilted-rmw-cyclonedds-cpp` |
| `ROS_DOMAIN_ID` | `0` |
| `CYCLONEDDS_URI` | `ros2_ws/config/cyclonedds_laptop.xml` |
| Env | `source setup_franka_env.sh` (repo root) sets all of the above |

`setup_franka_env.sh` also brings up the static IP via a **toggleable
NetworkManager profile** (`fr3-link`, created on first run with `autoconnect no`).
The wired port still does normal DHCP for other uses; the static IP is active only
while the profile is up. To revert manually: `nmcli connection down fr3-link`.
Override `FR3_IFACE` / `FR3_LAPTOP_IP` / `FR3_NM_PROFILE` before sourcing if the
hardware differs.

### NUC (`nu-crb`)

| | |
|---|---|
| OS | Ubuntu 22.04 Jammy |
| ROS2 | Humble |
| Laptop link | `enx00249b860356` = `10.0.0.2/24` |
| Robot link | `enp89s0` = `192.168.51.10/24`; FR3 at `192.168.51.20` |
| RMW | `rmw_cyclonedds_cpp` |
| `ROS_DOMAIN_ID` | unset → defaults to **0** |
| `CYCLONEDDS_URI` | `/home/franka/franka_ws/config/cyclonedds.xml` |

Bringup aliases (already configured on the NUC):

```bash
fr3-bringup        # ros2 launch franka_bringup franka.launch.py robot_ip:=192.168.51.20 arm_id:=fr3
fr3-arm-controller # ros2 run controller_manager spawner fr3_arm_controller \
                   #   -t joint_trajectory_controller/JointTrajectoryController \
                   #   --param-file .../franka_fr3_moveit_config/config/fr3_ros_controllers.yaml
```

### Shared DDS contract

Both machines must agree on all of:

- **RMW** `rmw_cyclonedds_cpp`.
- **`ROS_DOMAIN_ID` = 0** (the NUC leaves it unset, which is 0; the laptop sets it
  explicitly).
- **Unicast discovery only.** The NUC's `cyclonedds.xml` disables multicast and
  hardcodes the peer list `10.0.0.1` (laptop) and `10.0.0.2` (NUC). Therefore the
  **laptop must actually hold `10.0.0.1`** — there is no multicast fallback. If you
  use a different laptop IP, you must also edit the NUC's peer list.
- **Interface pinning.** Each side pins CycloneDDS to its NUC-link NIC
  (`enp0s31f6` on the laptop, `enx00249b860356` on the NUC) so discovery traffic
  doesn't leak onto WiFi or, later, the inference-server NIC.

`ros2_ws/config/cyclonedds_laptop.xml` is the laptop-side mirror of the NUC file.

### FR3 specifics

- **TF tree:** `base → fr3_link0 → … → fr3_link7 → fr3_link8 → fr3_hand → fr3_hand_tcp`.
  - Base frame: **`fr3_link0`**
  - EEF / tool frame: **`fr3_hand_tcp`** (tool center point, 0.1034 m past `fr3_hand`)
  - `policy_client_node` reads `base_frame` / `eef_frame` params (defaults above).
- **Gripper:** width on `/fr3_gripper/joint_states`; action servers
  `/fr3_gripper/{grasp,move,gripper_action,homing}`. (Wired into observations /
  execution in Phase 2; currently a `0.0` placeholder.)
- **Robot state:** `/franka_robot_state_broadcaster/current_pose` exposes the EEF
  pose as an alternative to the TF lookup, plus joint states / wrenches.

### Quick checks

```bash
# laptop, after `source setup_franka_env.sh` and with `fr3-bringup` up on the NUC:
ping 10.0.0.2
ros2 node list                                   # NUC nodes appear
ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp # live transform
```

## Running Demos & Inference

This brings up the **dummy** inference loop (no real checkpoint): the FR3 stack on
the NUC, the PolyUMI nodes + `policy_client_node` on the laptop, and the dummy
server (currently also on the laptop). At the end the client logs 8-vector actions
at 10 Hz, pulling the live EEF pose from the NUC's TF over DDS.

Start the three pieces in separate terminals, in this order.

**1. NUC — bring up the FR3** (enable FCI on the Desk UI first):

```bash
fr3-bringup          # franka_bringup, arm_id:=fr3, robot @ 192.168.51.20
fr3-arm-controller   # in a second terminal: spawn the joint-trajectory controller
```

**2. Laptop — dummy inference server** (its own terminal):

```bash
cd inference_server
uv run dummy-server   # FastAPI on 0.0.0.0:8000; oscillates X around HOME_POSE
```

**3. Laptop — PolyUMI ROS2 nodes + policy client** (another terminal):

```bash
source setup_franka_env.sh          # CycloneDDS + domain 0 + bring up the fr3-link NM profile
cd ros2_ws
source install/setup.bash           # (build first if needed: colcon build)
ros2 launch polyumi_ros2 inference_demo.launch.xml
# default inference_server_url is http://localhost:8000/predict_cartesian/
```

Confirm the loop is live: `policy_client_node` logs `action x=… y=… z=… grip=…`
at ~10 Hz, and Foxglove (`ws://localhost:8765`) shows the GoPro + FR3 TF. If the
client warns about TF lookups, re-check the [Quick checks](#quick-checks) above —
the NUC must be reachable and `fr3-bringup` running.
