"""Build and inspect pzarr working-format zarr stores."""

import dataclasses
import datetime as dt
import importlib.metadata
import logging
import pathlib
import subprocess
import wave

import cv2
import numcodecs
import numpy as np
import zarr
from imagecodecs.numcodecs import Jpegxl
from numcodecs import Blosc
from polyumi_pi.files.session import SessionFiles

from polyumi_ingest.pzarr.scene_files import SceneFiles
from polyumi_ingest.pzarr.version import PZARR_VERSION

numcodecs.register_codec(Jpegxl)

log = logging.getLogger('pzarr')

# effort=1: fastest encode; distance default (1.0) is perceptually lossless
_JPEGXL = Jpegxl(effort=1)
_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)


def _git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def _arr(grp: zarr.Group, path: str) -> zarr.Array:
    """Return a typed zarr.Array from a group by path; consolidates zarr's untyped __getitem__."""
    return grp[path]  # type: ignore[return-value]


def _finger_timestamps(video_dir: pathlib.Path, first_wall_ns: int) -> np.ndarray:
    """
    Return UTC-seconds float64 timestamps for each finger camera frame.

    SensorTimestamp in the CSV is a hardware monotonic counter. FrameWallClock
    from first_frame_metadata anchors it to absolute wall time.
    """
    csv_path = video_dir / 'video_timestamps.csv'
    rows = np.loadtxt(csv_path, delimiter=',', dtype=np.int64)
    sensor_ts = rows[:, 1]
    wall_ns = first_wall_ns + (sensor_ts - sensor_ts[0])
    return wall_ns.astype(np.float64) / 1e9


def _audio_timestamps(start_ns: int, n_samples: int, sample_rate: int) -> np.ndarray:
    """Return UTC-seconds float64 timestamps for each audio sample."""
    start_s = np.float64(start_ns) / 1e9
    return start_s + np.arange(n_samples, dtype=np.float64) / sample_rate


def _read_wav(audio_path: pathlib.Path) -> tuple[np.ndarray, int]:
    """Read a WAV file, return (samples as float32, sample_rate)."""
    with wave.open(str(audio_path), 'rb') as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    audio = np.frombuffer(raw, dtype=dtype)
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch)
    peak = max(abs(np.iinfo(dtype).min), np.iinfo(dtype).max)
    return audio.astype(np.float32) / peak, sr


def _write_episode(ep_grp: zarr.Group, session: SessionFiles, skip_gopro: bool) -> None:
    meta = session.metadata
    video_dir = session.path / 'video'
    audio_path = session.path / 'audio.wav'

    ts_grp = ep_grp.require_group('timestamps')
    ann_grp = ep_grp.require_group('annotations')

    # --- Finger camera ---
    frames = sorted(video_dir.glob('frame_*.jpg'))
    if not frames:
        raise RuntimeError(f'No finger frames found in {video_dir}')
    N = len(frames)

    sample = cv2.imdecode(np.frombuffer(frames[0].read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f'Failed to decode sample frame: {frames[0]}')
    H, W = sample.shape[:2]

    finger_grp = ep_grp.require_group('finger')
    frames_arr = finger_grp.zeros(
        name='frames',
        shape=(N, H, W, 3),
        dtype='uint8',
        chunks=(1, H, W, 3),
        compressor=_JPEGXL,
        zarr_format=2,
    )
    log.info(f'  Writing {N} finger frames ({H}x{W})...')
    for j, fp in enumerate(frames):
        raw = np.frombuffer(fp.read_bytes(), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning(f'  Failed to decode {fp.name}, skipping')
            continue
        frames_arr[j] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if meta.first_frame_metadata is None:
        raise RuntimeError(f'first_frame_metadata missing in {session.path / "metadata.json"}')
    first_wall_ns = meta.first_frame_metadata['FrameWallClock']
    finger_ts = _finger_timestamps(video_dir, first_wall_ns)
    ts_grp.create_array('finger', data=finger_ts, compressor=_BLOSC)

    # --- Audio ---
    if meta.audio_start_time_ns is None:
        raise RuntimeError(f'audio_start_time_ns missing in {session.path / "metadata.json"}')
    audio_data, sr = _read_wav(audio_path)
    audio_grp = ep_grp.require_group('audio')
    audio_grp.create_array('data', data=audio_data, compressor=_BLOSC)
    audio_ts = _audio_timestamps(meta.audio_start_time_ns, len(audio_data), sr)
    ts_grp.create_array('audio', data=audio_ts, compressor=_BLOSC)

    # --- GoPro ---
    if not skip_gopro:
        raise NotImplementedError('GoPro frame ingestion is not yet implemented. Use --skip-gopro to skip.')
    log.info('  Skipping GoPro frames (--skip-gopro).')

    # --- Annotations ---
    ann_grp.create_array('episode_start', data=np.array(finger_ts[0], dtype='float64'))
    ann_grp.create_array('episode_end', data=np.array(finger_ts[-1], dtype='float64'))


def build_scene_zarr(scene_path: pathlib.Path, skip_gopro: bool = False) -> pathlib.Path:
    """
    Build scene.zarr inside scene_path from processed session directories.

    Returns the path to the created zarr store.
    """
    scene = SceneFiles.from_path(scene_path)
    sessions = sorted(scene.sessions, key=lambda s: s.metadata.created_at)
    if not sessions:
        raise RuntimeError(f'No valid sessions found in {scene_path}')

    root = zarr.open_group(str(scene.zarr_path), mode='w', zarr_format=2)

    first_meta = sessions[0].metadata
    root.attrs.update(
        {
            'task': first_meta.task,
            'date': first_meta.created_at.date().isoformat(),
            'n_episodes': len(sessions),
            'location': None,
            'pipeline_version': importlib.metadata.version('polyumi_ingest'),
            'git_sha': _git_sha(),
            'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'alignment_refs': [],
            'pzarr_version': PZARR_VERSION,
        }
    )

    for i, session in enumerate(sessions):
        log.info(f'[{i + 1}/{len(sessions)}] Episode {i}: {session.path.name}')
        ep_grp = root.require_group(f'episode_{i}')
        _write_episode(ep_grp, session, skip_gopro)

    return scene.zarr_path


@dataclasses.dataclass
class EpisodeInfo:
    """Summary of one episode's arrays and timestamps extracted from scene.zarr."""

    index: int
    finger_shape: tuple | None  # type: ignore[type-arg]
    audio_shape: tuple | None  # type: ignore[type-arg]
    finger_ts_range: tuple[float, float] | None
    finger_ts_mean_delta_ms: float | None
    audio_ts_range: tuple[float, float] | None
    episode_start: float | None
    episode_end: float | None


@dataclasses.dataclass
class SceneZarrInfo:
    """Top-level summary of a scene.zarr store returned by inspect_scene_zarr."""

    zarr_path: pathlib.Path
    zarr_format: int
    tree: object
    attrs: dict  # type: ignore[type-arg]
    episodes: list[EpisodeInfo]


def inspect_scene_zarr(scene_path: pathlib.Path) -> SceneZarrInfo:
    """Open scene.zarr inside scene_path and extract summary info."""
    zarr_path = SceneFiles.resolve_zarr_path(scene_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

    root = zarr.open_group(str(zarr_path), mode='r')
    n_episodes = int(root.attrs.get('n_episodes', 0))  # type: ignore[arg-type]

    episodes = []
    for i in range(n_episodes):
        ep_key = f'episode_{i}'
        if ep_key not in root:
            continue
        ep = zarr.open_group(str(zarr_path / ep_key), mode='r')

        finger_ts_range: tuple[float, float] | None = None
        finger_mean_delta: float | None = None
        if 'timestamps/finger' in ep:
            ts: np.ndarray = _arr(ep, 'timestamps/finger')[:]  # type: ignore[assignment]
            finger_ts_range = (float(ts[0]), float(ts[-1]))
            finger_mean_delta = float(np.diff(ts).mean() * 1000) if len(ts) > 1 else None

        audio_ts_range: tuple[float, float] | None = None
        if 'timestamps/audio' in ep:
            ts = _arr(ep, 'timestamps/audio')[:]  # type: ignore[assignment]
            audio_ts_range = (float(ts[0]), float(ts[-1]))

        ep_start = (
            float(_arr(ep, 'annotations/episode_start')[()])  # type: ignore[arg-type]
            if 'annotations/episode_start' in ep
            else None
        )
        ep_end = (
            float(_arr(ep, 'annotations/episode_end')[()])  # type: ignore[arg-type]
            if 'annotations/episode_end' in ep
            else None
        )
        episodes.append(
            EpisodeInfo(
                index=i,
                finger_shape=_arr(ep, 'finger/frames').shape if 'finger/frames' in ep else None,
                audio_shape=_arr(ep, 'audio/data').shape if 'audio/data' in ep else None,
                finger_ts_range=finger_ts_range,
                finger_ts_mean_delta_ms=finger_mean_delta,
                audio_ts_range=audio_ts_range,
                episode_start=ep_start,
                episode_end=ep_end,
            )
        )

    return SceneZarrInfo(
        zarr_path=zarr_path,
        zarr_format=root.metadata.zarr_format,
        tree=root.tree(),
        attrs=dict(root.attrs),
        episodes=episodes,
    )


def read_frame(scene_path: pathlib.Path, episode: int = 0, frame: int = 0) -> np.ndarray:
    """Read a single frame from scene.zarr; returns (H, W, 3) uint8 RGB array."""
    zarr_path = SceneFiles.resolve_zarr_path(scene_path)
    root = zarr.open_group(str(zarr_path), mode='r')
    return _arr(root, f'episode_{episode}/finger/frames')[frame]  # type: ignore[return-value]
