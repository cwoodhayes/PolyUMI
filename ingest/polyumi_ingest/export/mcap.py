"""Export pzarr episodes to MCAP for visualization in Foxglove."""

import base64
import json
import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr
from mcap.writer import Writer

from polyumi_ingest.export.helpers import encode_frames_to_jpeg
from polyumi_ingest.pzarr.scene_files import SceneFiles
from polyumi_ingest.transforms import gripper_calib_transforms, transform_optitrack_pose

log = logging.getLogger('export.mcap')

# ── Foxglove JSON schemas ─────────────────────────────────────────────────────

_TIME = {
    'type': 'object',
    'properties': {
        'sec': {'type': 'integer', 'minimum': 0},
        'nsec': {'type': 'integer', 'minimum': 0, 'maximum': 999_999_999},
    },
}

_VEC3 = {
    'type': 'object',
    'properties': {
        'x': {'type': 'number'},
        'y': {'type': 'number'},
        'z': {'type': 'number'},
    },
}

_SCHEMA_COMPRESSED_IMAGE = json.dumps(
    {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'foxglove.CompressedImage',
        'type': 'object',
        'properties': {
            'timestamp': _TIME,
            'frame_id': {'type': 'string'},
            'data': {'type': 'string', 'contentEncoding': 'base64'},
            'format': {'type': 'string'},
        },
    }
).encode()

_SCHEMA_RAW_AUDIO = json.dumps(
    {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'foxglove.RawAudio',
        'type': 'object',
        'properties': {
            'timestamp': _TIME,
            'format': {'type': 'string'},
            'sample_rate': {'type': 'integer'},
            'number_of_channels': {'type': 'integer'},
            'data': {'type': 'string', 'contentEncoding': 'base64'},
        },
    }
).encode()

_SCHEMA_IMU = json.dumps(
    {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'foxglove.Imu',
        'type': 'object',
        'properties': {
            'timestamp': _TIME,
            'frame_id': {'type': 'string'},
            'orientation': {
                'type': 'object',
                'properties': {
                    'x': {'type': 'number'},
                    'y': {'type': 'number'},
                    'z': {'type': 'number'},
                    'w': {'type': 'number'},
                },
            },
            'orientation_covariance': {'type': 'array', 'items': {'type': 'number'}},
            'angular_velocity': _VEC3,
            'angular_velocity_covariance': {'type': 'array', 'items': {'type': 'number'}},
            'linear_acceleration': _VEC3,
            'linear_acceleration_covariance': {'type': 'array', 'items': {'type': 'number'}},
        },
    }
).encode()

_SCHEMA_POSE_IN_FRAME = json.dumps(
    {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'foxglove.PoseInFrame',
        'type': 'object',
        'properties': {
            'timestamp': _TIME,
            'frame_id': {'type': 'string'},
            'pose': {
                'type': 'object',
                'properties': {
                    'position': _VEC3,
                    'orientation': {
                        'type': 'object',
                        'properties': {
                            'x': {'type': 'number'},
                            'y': {'type': 'number'},
                            'z': {'type': 'number'},
                            'w': {'type': 'number'},
                        },
                    },
                },
            },
        },
    }
).encode()

_QUAT = {
    'type': 'object',
    'properties': {
        'x': {'type': 'number'},
        'y': {'type': 'number'},
        'z': {'type': 'number'},
        'w': {'type': 'number'},
    },
}

_SCHEMA_FRAME_TRANSFORM = json.dumps(
    {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'foxglove.FrameTransform',
        'type': 'object',
        'properties': {
            'timestamp': _TIME,
            'parent_frame_id': {'type': 'string'},
            'child_frame_id': {'type': 'string'},
            'translation': _VEC3,
            'rotation': _QUAT,
        },
    }
).encode()

_SCHEMA_LOCATION_FIX = json.dumps(
    {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'foxglove.LocationFix',
        'type': 'object',
        'properties': {
            'timestamp': _TIME,
            'frame_id': {'type': 'string'},
            'latitude': {'type': 'number'},
            'longitude': {'type': 'number'},
            'altitude': {'type': 'number'},
            'position_covariance': {'type': 'array', 'items': {'type': 'number'}},
            'position_covariance_type': {'type': 'integer'},
        },
    }
).encode()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ts_ns(t_s: float) -> int:
    """Convert UTC seconds to integer nanoseconds."""
    return round(t_s * 1e9)


def _foxglove_time(t_s: float) -> dict:  # type: ignore[type-arg]
    """Return a {sec, nsec} dict for Foxglove timestamp fields."""
    ns = _ts_ns(t_s)
    return {'sec': ns // 1_000_000_000, 'nsec': ns % 1_000_000_000}


def _b64(data: bytes) -> str:
    """Base64-encode bytes to an ASCII string for JSON embedding."""
    return base64.b64encode(data).decode('ascii')


# ── Channel registration ──────────────────────────────────────────────────────


def _register_channels(
    writer: Writer,
    *,
    has_gopro: bool,
    has_accel: bool,
    has_gyro: bool,
    has_gps: bool,
    has_gopro_audio: bool,
    has_optitrack: bool,
    has_slam: bool,
) -> dict[str, int]:
    """Register all schemas and channels; return {topic: channel_id}."""
    img_sid = writer.register_schema('foxglove.CompressedImage', 'jsonschema', _SCHEMA_COMPRESSED_IMAGE)
    aud_sid = writer.register_schema('foxglove.RawAudio', 'jsonschema', _SCHEMA_RAW_AUDIO)
    imu_sid = writer.register_schema('foxglove.Imu', 'jsonschema', _SCHEMA_IMU)
    gps_sid = writer.register_schema('foxglove.LocationFix', 'jsonschema', _SCHEMA_LOCATION_FIX)
    pose_sid = writer.register_schema('foxglove.PoseInFrame', 'jsonschema', _SCHEMA_POSE_IN_FRAME)
    tf_sid = writer.register_schema('foxglove.FrameTransform', 'jsonschema', _SCHEMA_FRAME_TRANSFORM)

    def ch(topic: str, sid: int) -> int:
        return writer.register_channel(topic=topic, message_encoding='json', schema_id=sid)

    channels: dict[str, int] = {
        '/finger/image': ch('/finger/image', img_sid),
        '/finger/piezo': ch('/finger/piezo', aud_sid),
        '/finger/air': ch('/finger/air', aud_sid),
        '/tf_static': ch('/tf_static', tf_sid),
    }
    if has_gopro:
        channels['/gopro/image'] = ch('/gopro/image', img_sid)
    if has_gopro_audio:
        channels['/gopro/audio'] = ch('/gopro/audio', aud_sid)
    if has_accel:
        channels['/gopro/accel'] = ch('/gopro/accel', imu_sid)
    if has_gyro:
        channels['/gopro/gyro'] = ch('/gopro/gyro', imu_sid)
    if has_gps:
        channels['/gopro/gps'] = ch('/gopro/gps', gps_sid)
    if has_optitrack:
        channels['/optitrack/pose'] = ch('/optitrack/pose', pose_sid)
    if has_slam:
        channels['/slam/pose'] = ch('/slam/pose', pose_sid)
    return channels


# ── Per-stream writers ────────────────────────────────────────────────────────


_ENCODE_BATCH = 64


def _write_video(
    writer: Writer,
    channel_id: int,
    frames_arr: zarr.Array,
    ts: np.ndarray,
    frame_id: str,
    quality: int,
) -> None:
    """Write video frames as CompressedImage messages, re-encoding JpegXL → JPEG."""
    n = len(ts)
    n_batches = (n + _ENCODE_BATCH - 1) // _ENCODE_BATCH
    log_every_batch = max(1, n_batches // 10)
    with ThreadPoolExecutor() as executor:
        for b, batch_start in enumerate(range(0, n, _ENCODE_BATCH)):
            if b % log_every_batch == 0:
                log.info(f'    {frame_id}: frame {batch_start}/{n}')
            batch_end = min(batch_start + _ENCODE_BATCH, n)
            jpegs = encode_frames_to_jpeg(np.asarray(frames_arr[batch_start:batch_end]), quality, executor)
            for j, jpeg in enumerate(jpegs):
                t_s = float(ts[batch_start + j])
                msg = json.dumps(
                    {
                        'timestamp': _foxglove_time(t_s),
                        'frame_id': frame_id,
                        'data': _b64(jpeg),
                        'format': 'jpeg',
                    }
                ).encode()
                writer.add_message(channel_id=channel_id, log_time=_ts_ns(t_s), data=msg, publish_time=_ts_ns(t_s))


def _write_audio(
    writer: Writer,
    channel_id: int,
    audio_data: np.ndarray,
    ts: np.ndarray,
    chunk_size: int,
) -> None:
    """
    Chunk audio into fixed-size blocks and write as RawAudio messages.

    pzarr stores float32 audio; this converts to int16 PCM with format 'pcm-s16',
    matching the format string used by the streaming node and recognized by
    Foxglove's audio panel.
    """
    n_samples = audio_data.shape[0]
    n_channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
    # Infer sample rate from timestamp spacing; fall back to 16 kHz if only one sample.
    sr = int(round(1.0 / float(ts[1] - ts[0]))) if len(ts) > 1 else 16_000

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = audio_data[start:end]
        pcm = (chunk * 32_767).clip(-32_768, 32_767).astype('<i2')
        t_s = float(ts[start])
        msg = json.dumps(
            {
                'timestamp': _foxglove_time(t_s),
                'format': 'pcm-s16',
                'sample_rate': sr,
                'number_of_channels': n_channels,
                'data': _b64(pcm.tobytes()),
            }
        ).encode()
        writer.add_message(channel_id=channel_id, log_time=_ts_ns(t_s), data=msg, publish_time=_ts_ns(t_s))


# -1 in the first covariance element is the ROS/Foxglove convention for "unknown".
_UNKNOWN_COV9 = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _write_imu(
    writer: Writer,
    channel_id: int,
    data_arr: np.ndarray,
    ts: np.ndarray,
    field: str,
) -> None:
    """
    Write accel or gyro samples as Imu messages.

    field must be 'accel' or 'gyro'. GoPro's native z-x-y column order is
    preserved as-is in the x, y, z fields of the Foxglove Imu message.
    """
    vec_key = 'linear_acceleration' if field == 'accel' else 'angular_velocity'
    cov_key = vec_key + '_covariance'
    for i in range(len(ts)):
        row = data_arr[i]
        t_s = float(ts[i])
        msg = json.dumps(
            {
                'timestamp': _foxglove_time(t_s),
                'frame_id': 'gopro_imu',
                vec_key: {'x': float(row[0]), 'y': float(row[1]), 'z': float(row[2])},
                cov_key: _UNKNOWN_COV9,
            }
        ).encode()
        writer.add_message(channel_id=channel_id, log_time=_ts_ns(t_s), data=msg, publish_time=_ts_ns(t_s))


def _write_gps(
    writer: Writer,
    channel_id: int,
    gps_arr: np.ndarray,
    ts: np.ndarray,
) -> None:
    """Write GPS samples as LocationFix messages using lat/lon/alt from the GPS5 columns."""
    for i in range(len(ts)):
        row = gps_arr[i]
        t_s = float(ts[i])
        msg = json.dumps(
            {
                'timestamp': _foxglove_time(t_s),
                'frame_id': 'gopro_gps',
                'latitude': float(row[0]),
                'longitude': float(row[1]),
                'altitude': float(row[2]),
            }
        ).encode()
        writer.add_message(channel_id=channel_id, log_time=_ts_ns(t_s), data=msg, publish_time=_ts_ns(t_s))


def _write_optitrack_poses(
    writer: Writer,
    channel_id: int,
    pose_arr: np.ndarray,
    ts: np.ndarray,
    gripper_calib: dict,
) -> None:
    """Write OptiTrack rigid-body poses as foxglove.PoseInFrame messages."""
    T_gb_rb, T_gb_gp, T_o_w = gripper_calib_transforms(gripper_calib)

    for i in range(len(ts)):
        row = transform_optitrack_pose(pose_arr[i], T_gb_rb, T_gb_gp, T_o_w)
        t_s = float(ts[i])
        msg = json.dumps(
            {
                'timestamp': _foxglove_time(t_s),
                'frame_id': 'world',
                'pose': {
                    'position': {'x': float(row[0]), 'y': float(row[1]), 'z': float(row[2])},
                    'orientation': {
                        'x': float(row[3]),
                        'y': float(row[4]),
                        'z': float(row[5]),
                        'w': float(row[6]),
                    },
                },
            }
        ).encode()
        writer.add_message(channel_id=channel_id, log_time=_ts_ns(t_s), data=msg, publish_time=_ts_ns(t_s))


def _write_static_transform(
    writer: Writer,
    channel_id: int,
    t_s: float,
    parent: str,
    child: str,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> None:
    """Write a single foxglove.FrameTransform message on /tf_static."""
    msg = json.dumps(
        {
            'timestamp': _foxglove_time(t_s),
            'parent_frame_id': parent,
            'child_frame_id': child,
            'translation': {'x': translation[0], 'y': translation[1], 'z': translation[2]},
            'rotation': {'x': rotation[0], 'y': rotation[1], 'z': rotation[2], 'w': rotation[3]},
        }
    ).encode()
    writer.add_message(channel_id=channel_id, log_time=_ts_ns(t_s), data=msg, publish_time=_ts_ns(t_s))


def _write_slam_poses(
    writer: Writer,
    channel_id: int,
    poses: np.ndarray,
    ts: np.ndarray,
    frame_id: str,
) -> None:
    """Write SLAM poses as PoseInFrame messages, skipping lost (NaN) frames."""
    n = len(ts)
    if len(poses) != n:
        raise ValueError(f'SLAM pose/timestamp length mismatch: poses={len(poses)} ts={n}')

    valid_idx = np.nonzero(~np.isnan(poses[:, 0]))[0]
    if valid_idx.size == 0:
        log.info('  slam poses: all frames lost, nothing to write')
        return

    for i in valid_idx:
        t_s = float(ts[i])
        tx, ty, tz, qx, qy, qz, qw = poses[i].tolist()
        msg = json.dumps(
            {
                'timestamp': _foxglove_time(t_s),
                'frame_id': frame_id,
                'pose': {
                    'position': {'x': tx, 'y': ty, 'z': tz},
                    'orientation': {'x': qx, 'y': qy, 'z': qz, 'w': qw},
                },
            }
        ).encode()
        writer.add_message(channel_id=channel_id, log_time=_ts_ns(t_s), data=msg, publish_time=_ts_ns(t_s))

    log.info(f'  slam poses: wrote {valid_idx.size}/{n} (lost {n - valid_idx.size})')


# ── Public API ────────────────────────────────────────────────────────────────


def export_episode_to_mcap(
    ep_grp: zarr.Group,
    output_path: pathlib.Path,
    jpeg_quality: int = 85,
    audio_chunk_size: int = 4096,
    root_grp: zarr.Group | None = None,
) -> None:
    """Write one pzarr episode group to an MCAP file at output_path."""
    has_gopro = 'gopro/frames' in ep_grp
    has_gopro_audio = 'gopro/audio' in ep_grp
    has_accel = 'gopro/accl' in ep_grp
    has_gyro = 'gopro/gyro' in ep_grp
    has_gps = 'gopro/gps' in ep_grp
    has_optitrack = root_grp is not None and 'optitrack/pose' in root_grp
    has_slam = 'gopro/slam_poses' in ep_grp
    has_time_sync = (
        'annotations/time_sync' in ep_grp
        and 'gopro_to_finger_offset_s' in ep_grp['annotations/time_sync'].attrs  # type: ignore[index]
    )

    # gopro_to_finger_offset_s = gopro_time - finger_time, so subtract it
    # from gopro timestamps to bring them into the finger (Pi) time domain.
    gopro_ts_shift = 0.0
    if has_time_sync:
        gopro_ts_shift = -float(ep_grp['annotations/time_sync']
                                .attrs['gopro_to_finger_offset_s'])  # type: ignore[index]
        log.info(f'  time sync: shifting gopro timestamps by {gopro_ts_shift:+.6f}s')

    def _gopro_ts(key: str) -> np.ndarray:
        ts: np.ndarray = ep_grp[f'timestamps/{key}'][:]  # type: ignore[index]
        return ts + gopro_ts_shift if gopro_ts_shift != 0.0 else ts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        writer = Writer(f)
        writer.start(profile='', library='polyumi_ingest')
        try:
            ch = _register_channels(
                writer,
                has_gopro=has_gopro,
                has_gopro_audio=has_gopro_audio,
                has_accel=has_accel,
                has_gyro=has_gyro,
                has_gps=has_gps,
                has_optitrack=has_optitrack,
                has_slam=has_slam,
            )

            # Static transform: slam → world.  Use the computed T_ws if available,
            # otherwise fall back to identity (slam and world share the same origin).
            if has_slam:
                t0 = float(ep_grp['timestamps/finger'][0])  # type: ignore
                t_ws_attrs = root_grp.attrs.get('slam_to_world_transform') if root_grp is not None else None
                if isinstance(t_ws_attrs, dict):
                    _write_static_transform(
                        writer,
                        ch['/tf_static'],
                        t0,
                        parent='world',
                        child='slam',
                        translation=tuple(t_ws_attrs['translation']),  # type: ignore[arg-type]
                        rotation=tuple(t_ws_attrs['rotation']),  # type: ignore[arg-type]
                    )
                else:
                    _write_static_transform(writer, ch['/tf_static'], t0, parent='world', child='slam')

            if has_optitrack:
                assert root_grp is not None
                log.info('  optitrack poses...')
                gripper_calib = root_grp.attrs.get('gripper_calib')
                if not isinstance(gripper_calib, dict):
                    raise RuntimeError(
                        'gripper_calib not found in scene.zarr root attrs; rebuild the pzarr to embed the calibration.'
                    )
                ot_ts: np.ndarray = root_grp['optitrack/timestamps'][:]  # type: ignore[index]
                ot_poses: np.ndarray = root_grp['optitrack/pose'][:]  # type: ignore[index]
                # Slice to the episode time window when annotations are available.
                ann_attrs = ep_grp['annotations'].attrs if 'annotations' in ep_grp else {}
                ep_start = ann_attrs.get('episode_start')
                ep_end = ann_attrs.get('episode_end')
                if ep_start is not None and ep_end is not None:
                    mask = (ot_ts >= float(ep_start)) & (ot_ts <= float(ep_end))  # type: ignore
                    ot_ts = ot_ts[mask]
                    ot_poses = ot_poses[mask]
                _write_optitrack_poses(writer, ch['/optitrack/pose'], ot_poses, ot_ts, gripper_calib)

            log.info('  finger frames...')
            _write_video(
                writer,
                ch['/finger/image'],
                ep_grp['finger/frames'],  # type: ignore[index]
                ep_grp['timestamps/finger'][:],  # type: ignore[index]
                frame_id='finger',
                quality=jpeg_quality,
            )

            log.info('  finger piezo audio...')
            _write_audio(
                writer,
                ch['/finger/piezo'],
                ep_grp['finger/finger_piezo'][:],  # type: ignore[index]
                ep_grp['timestamps/finger_piezo'][:],  # type: ignore[index]
                chunk_size=audio_chunk_size,
            )

            log.info('  finger air audio...')
            _write_audio(
                writer,
                ch['/finger/air'],
                ep_grp['finger/finger_air'][:],  # type: ignore[index]
                ep_grp['timestamps/finger_air'][:],  # type: ignore[index]
                chunk_size=audio_chunk_size,
            )

            if has_gopro:
                log.info('  gopro frames...')
                _write_video(
                    writer,
                    ch['/gopro/image'],
                    ep_grp['gopro/frames'],  # type: ignore[index]
                    _gopro_ts('gopro'),
                    frame_id='gopro',
                    quality=jpeg_quality,
                )

            if has_gopro_audio:
                log.info('  gopro audio...')
                _write_audio(
                    writer,
                    ch['/gopro/audio'],
                    ep_grp['gopro/audio'][:],  # type: ignore[index]
                    _gopro_ts('gopro_audio'),
                    chunk_size=audio_chunk_size,
                )

            if has_accel:
                log.info('  accel...')
                _write_imu(
                    writer,
                    ch['/gopro/accel'],
                    ep_grp['gopro/accl'][:],  # type: ignore[index]
                    _gopro_ts('gopro_accl'),
                    field='accel',
                )

            if has_gyro:
                log.info('  gyro...')
                _write_imu(
                    writer,
                    ch['/gopro/gyro'],
                    ep_grp['gopro/gyro'][:],  # type: ignore[index]
                    _gopro_ts('gopro_gyro'),
                    field='gyro',
                )

            if has_gps:
                log.info('  gps...')
                _write_gps(
                    writer,
                    ch['/gopro/gps'],
                    ep_grp['gopro/gps'][:],  # type: ignore[index]
                    _gopro_ts('gopro_gps'),
                )

            if has_slam:
                log.info('  slam poses...')
                _write_slam_poses(
                    writer,
                    ch['/slam/pose'],
                    np.asarray(ep_grp['gopro/slam_poses'][:]),  # type: ignore[index]
                    _gopro_ts('gopro'),
                    frame_id='slam',
                )
        finally:
            writer.finish()


def export_scene_to_mcap(
    scene_path: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    episode: int | None = None,
    jpeg_quality: int = 85,
    audio_chunk_size: int = 4096,
) -> list[pathlib.Path]:
    """
    Export pzarr episodes from a scene to MCAP files, one file per episode.

    Returns the list of written .mcap paths.
    """
    zarr_path = SceneFiles.resolve_zarr_path(scene_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

    root = zarr.open_group(str(zarr_path), mode='r')
    n_episodes = int(root.attrs.get('n_episodes', 0))  # type: ignore[arg-type]

    out_dir = output_dir if output_dir is not None else zarr_path.parent
    ep_indices = [episode] if episode is not None else list(range(n_episodes))

    written: list[pathlib.Path] = []
    for ep_idx in ep_indices:
        ep_key = f'episode_{ep_idx}'
        if ep_key not in root:
            log.warning(f'Episode {ep_idx} not found in {zarr_path.name}, skipping.')
            continue
        ep_grp = zarr.open_group(str(zarr_path / ep_key), mode='r')
        out_path = out_dir / f'episode_{ep_idx}.mcap'
        log.info(f'Exporting episode {ep_idx} → {out_path}')
        export_episode_to_mcap(ep_grp, out_path, jpeg_quality, audio_chunk_size, root_grp=root)
        size_mb = out_path.stat().st_size / 1e6
        log.info(f'  Done ({size_mb:.1f} MB)')
        written.append(out_path)

    return written
