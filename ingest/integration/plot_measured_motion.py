"""
Plot the FR3's actual measured end-effector motion from a rosbag.

Reads /franka_robot_state_broadcaster/current_pose (the measured Cartesian EE pose,
read straight off libfranka's O_T_EE at the ~1kHz control-loop rate) and plots
position, velocity, and acceleration vs time. Unlike plot_action_chunks.py, which
only sees the discrete 10Hz target waypoints the policy produces, this is the
robot's actual physical motion — the only way to see the effect of anything that
happens purely inside the controller's real-time interpolation, such as the
choice of interpolant between waypoints.

Optionally overlays /franka_robot_state_broadcaster/desired_end_effector_twist,
the controller's own desired-velocity signal, if present in the bag.

This script needs rclpy/rosbag2_py/message types, which live in ROS's
interpreter, not the uv venv:

    bash -c 'source /opt/ros/kilted/setup.bash \
      && /usr/bin/python3 ingest/integration/plot_measured_motion.py <bag_path> [<bag_path> ...]'
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

POSE_TOPIC = '/franka_robot_state_broadcaster/current_pose'
TWIST_TOPIC = '/franka_robot_state_broadcaster/desired_end_effector_twist'


def _open_reader(bag_path: pathlib.Path) -> tuple[rosbag2_py.SequentialReader, dict[str, str]]:
    """Open a rosbag2 reader, autodetecting the storage format from metadata.yaml."""
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, topic_types


def _read_pose_and_twist(
    bag_path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read measured pose (and desired twist, if present) into time/position arrays."""
    reader, topic_types = _open_reader(bag_path)
    if POSE_TOPIC not in topic_types:
        raise SystemExit(f'{POSE_TOPIC} not found in {bag_path}. Topics present: {sorted(topic_types)}')
    pose_type = get_message(topic_types[POSE_TOPIC])
    twist_type = get_message(topic_types[TWIST_TOPIC]) if TWIST_TOPIC in topic_types else None

    pose_t, pose_xyz = [], []
    twist_t, twist_xyz = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == POSE_TOPIC:
            msg = deserialize_message(data, pose_type)
            pose_t.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            p = msg.pose.position
            pose_xyz.append((p.x, p.y, p.z))
        elif topic == TWIST_TOPIC:
            msg = deserialize_message(data, twist_type)
            twist_t.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            lin = msg.twist.linear
            twist_xyz.append((lin.x, lin.y, lin.z))

    pose_t_arr = np.asarray(pose_t)
    pose_xyz_arr = np.asarray(pose_xyz)
    if not twist_t:
        return pose_t_arr, pose_xyz_arr, None, None
    return pose_t_arr, pose_xyz_arr, np.asarray(twist_t), np.asarray(twist_xyz)


def _central_diff(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Central-difference derivative of x(t), same length as x (edges use one-sided diffs)."""
    d = np.empty_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])[:, None]
    d[0] = (x[1] - x[0]) / (t[1] - t[0])
    d[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    return d


def _resample_uniform(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample x(t) onto a uniform grid at the data's own median rate.

    The publisher's timestamps jitter by a fraction of a millisecond around
    their nominal 1kHz spacing; dividing by an occasional near-zero raw `dt`
    in a plain finite difference turns that jitter into spurious enormous
    velocity/acceleration/jerk spikes that have nothing to do with real
    motion. Resampling onto an evenly-spaced grid before differentiating
    removes that artifact instead of trying to filter it out after the fact.
    """
    dt = np.median(np.diff(t))
    t_uniform = np.arange(t[0], t[-1], dt)
    x_uniform = np.column_stack([np.interp(t_uniform, t, x[:, i]) for i in range(x.shape[1])])
    return t_uniform, x_uniform


def plot_motion(bag_path: pathlib.Path) -> None:
    """Plot measured position, velocity, and acceleration for one bag."""
    t, pos, twist_t, twist_xyz = _read_pose_and_twist(bag_path)
    t, pos = _resample_uniform(t, pos)
    t0 = t[0]
    ts = t - t0

    vel = _central_diff(t, pos)
    acc = _central_diff(t, vel)
    jerk = _central_diff(t, acc)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 11))
    labels = ['x', 'y', 'z']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, label in enumerate(labels):
        axes[0].plot(ts, pos[:, i], color=colors[i], linewidth=0.8, label=label)
        axes[1].plot(ts, vel[:, i], color=colors[i], linewidth=0.6, label=label)
        axes[2].plot(ts, acc[:, i], color=colors[i], linewidth=0.4, label=label)
        axes[3].plot(ts, jerk[:, i], color=colors[i], linewidth=0.3, alpha=0.8, label=label)

    if twist_t is not None:
        twist_ts = twist_t - t0
        for i, label in enumerate(labels):
            axes[1].plot(twist_ts, twist_xyz[:, i], color=colors[i], linewidth=0.6, linestyle='--', alpha=0.5)
        axes[1].plot([], [], color='gray', linestyle='--', label='desired (dashed)')

    axes[0].set_ylabel('measured position (m)')
    axes[1].set_ylabel('velocity (m/s)\nsolid=measured, dashed=desired')
    axes[2].set_ylabel('acceleration (m/s\N{SUPERSCRIPT TWO})')
    axes[3].set_ylabel('jerk (m/s\N{SUPERSCRIPT THREE})')
    axes[3].set_xlabel('time (s)')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=4)

    jerk_mag = np.linalg.norm(jerk, axis=1)
    fig.suptitle(
        f'{bag_path.name}: measured EE motion  '
        f'(|jerk| rms={np.sqrt(np.mean(jerk_mag**2)):.1f}, p99={np.percentile(jerk_mag, 99):.1f}, '
        f'max={jerk_mag.max():.1f} m/s\N{SUPERSCRIPT THREE})'
    )
    fig.tight_layout()
    out_path = bag_path.with_name(bag_path.name.rstrip('/') + '_measured_motion.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'{bag_path}: {len(t)} pose samples over {ts[-1]:.1f}s')
    print(
        f'  |jerk| rms={np.sqrt(np.mean(jerk_mag**2)):.1f}  p50={np.percentile(jerk_mag, 50):.1f}  '
        f'p99={np.percentile(jerk_mag, 99):.1f}  max={jerk_mag.max():.1f} m/s^3'
    )
    print(f'saved {out_path}')


def main() -> None:
    """Parse CLI args and plot each requested bag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_paths', type=pathlib.Path, nargs='+')
    args = parser.parse_args()
    for bag_path in args.bag_paths:
        plot_motion(bag_path)


if __name__ == '__main__':
    main()
