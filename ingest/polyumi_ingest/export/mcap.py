"""Export pzarr episodes to MCAP for visualization in Foxglove."""

import base64
import json
import logging
import pathlib

import cv2
import numpy as np
import zarr
from mcap.writer import Writer

from polyumi_ingest.pzarr.scene_files import SceneFiles

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
    return int(t_s * 1e9)


def _foxglove_time(t_s: float) -> dict:  # type: ignore[type-arg]
    """Return a {sec, nsec} dict for Foxglove timestamp fields."""
    ns = _ts_ns(t_s)
    return {'sec': ns // 1_000_000_000, 'nsec': ns % 1_000_000_000}


def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    """Encode a (H, W, 3) uint8 BGR array to JPEG bytes."""
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return buf.tobytes()


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
) -> dict[str, int]:
    """Register all schemas and channels; return {topic: channel_id}."""
    img_sid = writer.register_schema('foxglove.CompressedImage', 'jsonschema', _SCHEMA_COMPRESSED_IMAGE)
    aud_sid = writer.register_schema('foxglove.RawAudio', 'jsonschema', _SCHEMA_RAW_AUDIO)
    imu_sid = writer.register_schema('foxglove.Imu', 'jsonschema', _SCHEMA_IMU)
    gps_sid = writer.register_schema('foxglove.LocationFix', 'jsonschema', _SCHEMA_LOCATION_FIX)

    def ch(topic: str, sid: int) -> int:
        return writer.register_channel(topic=topic, message_encoding='json', schema_id=sid)

    channels: dict[str, int] = {
        '/finger/image': ch('/finger/image', img_sid),
        '/audio': ch('/audio', aud_sid),
    }
    if has_gopro:
        channels['/gopro/image'] = ch('/gopro/image', img_sid)
    if has_accel:
        channels['/gopro/accel'] = ch('/gopro/accel', imu_sid)
    if has_gyro:
        channels['/gopro/gyro'] = ch('/gopro/gyro', imu_sid)
    if has_gps:
        channels['/gopro/gps'] = ch('/gopro/gps', gps_sid)
    return channels


# ── Per-stream writers ────────────────────────────────────────────────────────


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
    log_every = max(1, n // 10)
    for i in range(n):
        if i % log_every == 0:
            log.info(f'    {frame_id}: frame {i}/{n}')
        jpeg = _encode_jpeg(frames_arr[i], quality)
        t_s = float(ts[i])
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

    pzarr stores float32 audio; this converts to pcm-s16le for Foxglove compatibility,
    matching the format sent by the streaming node.
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
                'format': 'pcm-s16le',
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


# ── Public API ────────────────────────────────────────────────────────────────


def export_episode_to_mcap(
    ep_grp: zarr.Group,
    output_path: pathlib.Path,
    jpeg_quality: int = 85,
    audio_chunk_size: int = 4096,
) -> None:
    """Write one pzarr episode group to an MCAP file at output_path."""
    has_gopro = 'gopro/frames' in ep_grp
    has_accel = 'gopro/accl' in ep_grp
    has_gyro = 'gopro/gyro' in ep_grp
    has_gps = 'gopro/gps' in ep_grp

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        writer = Writer(f)
        writer.start(profile='', library='polyumi_ingest')
        try:
            ch = _register_channels(
                writer,
                has_gopro=has_gopro,
                has_accel=has_accel,
                has_gyro=has_gyro,
                has_gps=has_gps,
            )

            log.info('  finger frames...')
            _write_video(
                writer,
                ch['/finger/image'],
                ep_grp['finger/frames'],  # type: ignore[index]
                ep_grp['timestamps/finger'][:],  # type: ignore[index]
                frame_id='finger',
                quality=jpeg_quality,
            )

            log.info('  audio...')
            _write_audio(
                writer,
                ch['/audio'],
                ep_grp['audio/data'][:],  # type: ignore[index]
                ep_grp['timestamps/audio'][:],  # type: ignore[index]
                chunk_size=audio_chunk_size,
            )

            if has_gopro:
                log.info('  gopro frames...')
                _write_video(
                    writer,
                    ch['/gopro/image'],
                    ep_grp['gopro/frames'],  # type: ignore[index]
                    ep_grp['timestamps/gopro'][:],  # type: ignore[index]
                    frame_id='gopro',
                    quality=jpeg_quality,
                )

            if has_accel:
                log.info('  accel...')
                _write_imu(
                    writer,
                    ch['/gopro/accel'],
                    ep_grp['gopro/accl'][:],  # type: ignore[index]
                    ep_grp['timestamps/gopro_accl'][:],  # type: ignore[index]
                    field='accel',
                )

            if has_gyro:
                log.info('  gyro...')
                _write_imu(
                    writer,
                    ch['/gopro/gyro'],
                    ep_grp['gopro/gyro'][:],  # type: ignore[index]
                    ep_grp['timestamps/gopro_gyro'][:],  # type: ignore[index]
                    field='gyro',
                )

            if has_gps:
                log.info('  gps...')
                _write_gps(
                    writer,
                    ch['/gopro/gps'],
                    ep_grp['gopro/gps'][:],  # type: ignore[index]
                    ep_grp['timestamps/gopro_gps'][:],  # type: ignore[index]
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
        export_episode_to_mcap(ep_grp, out_path, jpeg_quality, audio_chunk_size)
        size_mb = out_path.stat().st_size / 1e6
        log.info(f'  Done ({size_mb:.1f} MB)')
        written.append(out_path)

    return written
