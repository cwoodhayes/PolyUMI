"""
Plot end-effector action-chunk trajectories recorded in a rosbag.

Reads /polyumi/target_poses_traj (falling back to /polyumi/target_poses_preview)
from a ROS2 bag and plots x/y/z vs time, one color per action chunk, to help
spot discontinuities between chunks or noise within a chunk during diffusion
policy inference debugging.

This script needs rclpy/rosbag2_py/message types, which live in ROS's
interpreter, not the uv venv:

    bash -c 'source /opt/ros/kilted/setup.bash && source ros2_ws/install/setup.bash \
      && /usr/bin/python3 ingest/integration/plot_action_chunks.py <bag_path> [<bag_path> ...]'

/polyumi/target_poses_traj (trajectory_msgs/MultiDOFJointTrajectory) is only
published when the policy client runs with execute_motion:=true, and carries
an absolute per-waypoint schedule (header.stamp + points[i].time_from_start).
/polyumi/target_poses_preview (geometry_msgs/PoseArray) is always published,
but has no per-waypoint timestamp, so its waypoint times are approximated
from the bag's receive time plus a uniform --action-dt spacing.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

TRAJ_TOPIC = '/polyumi/target_poses_traj'
PREVIEW_TOPIC = '/polyumi/target_poses_preview'
# Matches control_hz: 10 in ros2_ws/src/polyumi_ros2/config/inference.yaml.
DEFAULT_ACTION_DT_S = 0.1

Chunk = tuple[list[float], list[tuple[float, float, float]]]


def _open_reader(bag_path: pathlib.Path) -> tuple[rosbag2_py.SequentialReader, dict[str, str]]:
    """
    Open a rosbag2 reader and return it with a topic-name -> type-name map.

    Passes an empty storage_id so rosbag2 autodetects the format (sqlite3 or
    mcap) from the bag's own metadata.yaml.
    """
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, topic_types


def _read_traj_chunks(reader: rosbag2_py.SequentialReader, msg_type: type) -> list[Chunk]:
    """Read MultiDOFJointTrajectory messages into (times_s, xyz) per chunk."""
    chunks: list[Chunk] = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != TRAJ_TOPIC:
            continue
        msg = deserialize_message(data, msg_type)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        times = []
        xyz = []
        for point in msg.points:
            t = stamp + point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            translation = point.transforms[0].translation
            times.append(t)
            xyz.append((translation.x, translation.y, translation.z))
        chunks.append((times, xyz))
    return chunks


def _read_preview_chunks(reader: rosbag2_py.SequentialReader, msg_type: type, action_dt_s: float) -> list[Chunk]:
    """Read PoseArray messages, approximating per-waypoint times via uniform spacing."""
    chunks: list[Chunk] = []
    while reader.has_next():
        topic, data, recv_time_ns = reader.read_next()
        if topic != PREVIEW_TOPIC:
            continue
        msg = deserialize_message(data, msg_type)
        t0 = recv_time_ns * 1e-9
        times = [t0 + i * action_dt_s for i in range(len(msg.poses))]
        xyz = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        chunks.append((times, xyz))
    return chunks


def _executed_count(chunk: Chunk, next_chunk: Chunk | None) -> int:
    """
    Count how many of a chunk's waypoints run before the next chunk preempts it.

    The streaming controller splices each new trajectory in by absolute time, so
    everything in `chunk` at or after `next_chunk`'s first waypoint time is
    overwritten before it ever executes. A chunk with no successor (the last one
    in the bag) is treated as fully executed.
    """
    if next_chunk is None:
        return len(chunk[0])
    cutoff = next_chunk[0][0]
    return sum(1 for t in chunk[0] if t < cutoff)


def plot_chunks(chunks: list[Chunk], title: str, out_path: pathlib.Path) -> None:
    """
    Plot x, y, z vs time for each action chunk, annotating what actually ran.

    Each chunk is drawn in two styles: a solid, opaque line for the waypoints
    that executed before the next chunk preempted them, and a faint dashed line
    for the remaining "planned but never executed" tail.
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    t0 = chunks[0][0][0]
    cmap = plt.get_cmap('viridis')
    n = len(chunks)
    executed_counts = []
    for i, (times, xyz) in enumerate(chunks):
        next_chunk = chunks[i + 1] if i + 1 < n else None
        n_exec = _executed_count((times, xyz), next_chunk)
        executed_counts.append(n_exec)
        color = cmap(i / max(n - 1, 1))
        ts = [t - t0 for t in times]
        # The tail line starts one point early (index n_exec - 1) so it visibly
        # connects to the executed line instead of leaving a gap.
        tail_start = max(n_exec - 1, 0)
        for axis_idx in range(3):
            vals = [p[axis_idx] for p in xyz]
            axes[axis_idx].plot(
                ts[:n_exec], vals[:n_exec], color=color, marker='o', markersize=3, linewidth=1.5, zorder=3
            )
            axes[axis_idx].plot(
                ts[tail_start:],
                vals[tail_start:],
                color=color,
                marker='o',
                markersize=1.5,
                linewidth=0.6,
                linestyle='--',
                alpha=0.35,
                zorder=1,
            )
    for axis_idx, label in enumerate(['x', 'y', 'z']):
        axes[axis_idx].set_ylabel(f'{label} (m)')
        axes[axis_idx].grid(True, alpha=0.3)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(
        f'{title} ({n} chunks, colored early\N{RIGHTWARDS ARROW}late; '
        'solid = executed before next chunk preempted it, dashed = never executed)'
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # The last chunk's count isn't comparable (nothing preempts it), so exclude it.
    comparable = executed_counts[:-1] if len(executed_counts) > 1 else executed_counts
    lengths = [len(c[0]) for c in chunks]
    print(
        f'chunk length: mean={sum(lengths) / len(lengths):.1f} (min={min(lengths)}, max={max(lengths)}); '
        f'executed before preemption: mean={sum(comparable) / len(comparable):.1f} '
        f'(min={min(comparable)}, max={max(comparable)})'
    )
    print(f'saved {out_path}')


def _load_chunks(bag_path: pathlib.Path, topic_choice: str | None, action_dt_s: float) -> tuple[list[Chunk], str]:
    """Load chunks from one bag, preferring the traj topic unless overridden."""
    reader, topic_types = _open_reader(bag_path)

    if topic_choice != 'preview' and TRAJ_TOPIC in topic_types:
        msg_type = get_message(topic_types[TRAJ_TOPIC])
        return _read_traj_chunks(reader, msg_type), TRAJ_TOPIC
    if PREVIEW_TOPIC in topic_types:
        msg_type = get_message(topic_types[PREVIEW_TOPIC])
        return _read_preview_chunks(reader, msg_type, action_dt_s), PREVIEW_TOPIC
    raise SystemExit(
        f'Neither {TRAJ_TOPIC} nor {PREVIEW_TOPIC} found in {bag_path}. Topics present: {sorted(topic_types)}'
    )


def main() -> None:
    """Parse CLI args and plot each requested bag."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag_paths', type=pathlib.Path, nargs='+')
    parser.add_argument(
        '--topic',
        choices=['traj', 'preview'],
        default=None,
        help='Force which topic to read; default prefers traj, falling back to preview.',
    )
    parser.add_argument(
        '--action-dt',
        type=float,
        default=DEFAULT_ACTION_DT_S,
        help='Waypoint spacing (s), used only when reading the preview topic.',
    )
    args = parser.parse_args()

    for bag_path in args.bag_paths:
        chunks, topic = _load_chunks(bag_path, args.topic, args.action_dt)
        if not chunks:
            print(f'no messages found on {topic} in {bag_path}, skipping')
            continue
        timing_note = 'exact' if topic == TRAJ_TOPIC else f'approx, dt={args.action_dt}s'
        print(f'{bag_path}: read {len(chunks)} chunks from {topic} ({timing_note} timing)')
        out_path = bag_path.with_name(bag_path.name.rstrip('/') + '_action_chunks.png')
        plot_chunks(chunks, title=f'{bag_path.name} [{topic}]', out_path=out_path)


if __name__ == '__main__':
    main()
