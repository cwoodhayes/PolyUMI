"""Build and inspect pzarr working-format zarr stores."""

import dataclasses
import datetime as dt
import importlib.metadata
import json
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

from polyumi_ingest.gopro_fetch import _recording_start_time
from polyumi_ingest.gpmf_parse import extract_gpmf_binary, parse_imu
from polyumi_ingest.pzarr.optitrack import find_optitrack_csv, write_optitrack
from polyumi_ingest.pzarr.scene_files import GOPRO_MP4, SceneFiles
from polyumi_ingest.pzarr.version import PZARR_VERSION
from polyumi_ingest.video_helpers import write_frames_to_zarr

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


def arr(grp: zarr.Group, path: str) -> zarr.Array:
    """Return a typed zarr.Array from a group by path; consolidates zarr's untyped __getitem__."""
    return grp[path]  # type: ignore[return-value]


def grp(grp: zarr.Group, path: str) -> zarr.Group:
    """Narrow Group.__getitem__'s Array|Group return to Group for type checkers."""
    return grp[path]  # type: ignore[return-value]


def _finger_timestamps(video_dir: pathlib.Path, first_wall_ns: int) -> np.ndarray:
    """
    Return UTC-seconds float64 timestamps for each finger camera frame.

    SensorTimestamp in the CSV is a hardware monotonic counter. FrameWallClock
    from first_frame_metadata anchors it to absolute wall time.
    """
    csv_path = video_dir / 'video_timestamps.csv'
    rows = np.loadtxt(csv_path, delimiter=',', dtype=np.int64)
    rows = np.atleast_2d(rows)
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
    if sw == 1:
        # 8-bit WAV PCM is unsigned (0–255, silence at 128)
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw in (2, 4):
        dtype = np.int16 if sw == 2 else np.int32
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / float(-np.iinfo(dtype).min)
    else:
        raise ValueError(f'Unsupported WAV sample width {sw} bytes in {audio_path.name}')
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch)
    return audio, sr


def _write_gopro_imu(
    ep_grp: zarr.Group,
    gopro_path: pathlib.Path,
    recording_start_s: float,
    duration_s: float,
) -> None:
    """
    Extract GPMF IMU/GPS from gopro.mp4 and write into ep_grp.

    Writes to gopro/{accl,gyro,gps} and timestamps/gopro_{accl,gyro,gps}.
    Timestamps are uniformly spaced across the recording: GoPro samples each
    IMU sensor at a constant hardware rate, so this matches reality within the
    ~1 ms jitter of the GPMF container boundaries.
    """
    gpmf_data = extract_gpmf_binary(gopro_path)
    if gpmf_data is None:
        return

    imu = parse_imu(gpmf_data)
    gopro_grp = ep_grp.require_group('gopro')
    ts_grp = ep_grp.require_group('timestamps')

    def _uniform_ts(n: int) -> np.ndarray:
        return recording_start_s + np.arange(n, dtype=np.float64) / (n / duration_s)

    if imu.accl is not None:
        n = len(imu.accl)
        gopro_grp.create_array('accl', data=imu.accl, compressor=_BLOSC)
        ts_grp.create_array('gopro_accl', data=_uniform_ts(n), compressor=_BLOSC)
        log.info(f'  GoPro ACCL: {n} samples (~{n / duration_s:.0f} Hz)')

    if imu.gyro is not None:
        n = len(imu.gyro)
        gopro_grp.create_array('gyro', data=imu.gyro, compressor=_BLOSC)
        ts_grp.create_array('gopro_gyro', data=_uniform_ts(n), compressor=_BLOSC)
        log.info(f'  GoPro GYRO: {n} samples (~{n / duration_s:.0f} Hz)')

    if imu.gps is not None:
        n = len(imu.gps)
        gopro_grp.create_array('gps', data=imu.gps, compressor=_BLOSC)
        ts_grp.create_array('gopro_gps', data=_uniform_ts(n), compressor=_BLOSC)
        log.info(f'  GoPro GPS:  {n} samples (~{n / duration_s:.0f} Hz)')


def _write_gopro_audio(
    ep_grp: zarr.Group,
    gopro_path: pathlib.Path,
    recording_start_s: float,
) -> None:
    """
    Extract audio from gopro.mp4 and write into ep_grp.

    Writes to gopro/audio and timestamps/gopro_audio. Uses ffprobe to get the
    native sample rate and channel count, then ffmpeg to extract raw float32 PCM.
    """
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(gopro_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(f'ffprobe failed on {gopro_path.name}: {exc}')
        return

    audio_info = None
    for s in json.loads(probe.stdout).get('streams', []):
        if s.get('codec_type') == 'audio':
            audio_info = s
            break

    if audio_info is None:
        log.info(f'  No audio stream in {gopro_path.name}')
        return

    sr = int(audio_info.get('sample_rate', 48000))
    n_ch = int(audio_info.get('channels', 2))
    duration_s = float(audio_info.get('duration', 0) or 0)
    expected_mb = duration_s * sr * n_ch * 4 / 1e6
    log.info(f'  GoPro audio: {sr} Hz {n_ch}ch ~{duration_s:.1f}s → ~{expected_mb:.0f} MB RAM')

    try:
        result = subprocess.run(
            ['ffmpeg', '-i', str(gopro_path), '-vn', '-f', 'f32le', '-ar', str(sr), '-ac', str(n_ch), 'pipe:1'],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(f'ffmpeg audio extraction failed on {gopro_path.name}: {exc}')
        return

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch)
    n_samples = audio.shape[0]

    gopro_grp = ep_grp.require_group('gopro')
    ts_grp = ep_grp.require_group('timestamps')
    gopro_grp.create_array('audio', data=audio, compressor=_BLOSC)
    ts_grp.create_array(
        'gopro_audio',
        data=recording_start_s + np.arange(n_samples, dtype=np.float64) / sr,
        compressor=_BLOSC,
    )
    log.info(f'  GoPro audio: {n_samples} samples, {n_ch}ch @ {sr} Hz ({n_samples / sr:.1f}s)')


def _write_gopro_frames(ep_grp: zarr.Group, gopro_path: pathlib.Path) -> None:
    """Decode gopro.mp4 and write frames, timestamps, and IMU into ep_grp."""
    cap = cv2.VideoCapture(str(gopro_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open GoPro video: {gopro_path}')
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if n_frames <= 0 or W <= 0 or H <= 0 or fps <= 0:
            raise RuntimeError(f'Could not read video properties from {gopro_path}')

        recording_start_s = _recording_start_time(gopro_path).timestamp()

        gopro_grp = ep_grp.require_group('gopro')
        frames_arr = gopro_grp.zeros(
            name='frames',
            shape=(n_frames, H, W, 3),
            dtype='uint8',
            chunks=(1, H, W, 3),
            compressor=_JPEGXL,
            zarr_format=2,
        )

        log.info(f'  Writing {n_frames} GoPro frames ({W}x{H}, {fps:.1f} fps)...')
        n_written = write_frames_to_zarr(gopro_path, frames_arr)

        if n_written < n_frames:
            frames_arr.resize((n_written, H, W, 3))
            n_frames = n_written

        gopro_ts = recording_start_s + np.arange(n_frames, dtype=np.float64) / fps
        ts_grp = ep_grp.require_group('timestamps')
        ts_grp.create_array('gopro', data=gopro_ts, compressor=_BLOSC)
    finally:
        cap.release()

    _write_gopro_imu(ep_grp, gopro_path, recording_start_s, n_frames / fps)
    _write_gopro_audio(ep_grp, gopro_path, recording_start_s)


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
    n_written = write_frames_to_zarr(frames, frames_arr)
    if n_written < N:
        frames_arr.resize((n_written, H, W, 3))

    if meta.first_frame_metadata is None:
        raise RuntimeError(f'first_frame_metadata missing in {session.path / "metadata.json"}')
    first_wall_ns = meta.first_frame_metadata['FrameWallClock']
    finger_ts = _finger_timestamps(video_dir, first_wall_ns)[:n_written]
    ts_grp.create_array('finger', data=finger_ts, compressor=_BLOSC)

    # --- Finger audio (L=piezo, R=air mic) ---
    if meta.audio_start_time_ns is None:
        raise RuntimeError(f'audio_start_time_ns missing in {session.path / "metadata.json"}')
    audio_data, sr = _read_wav(audio_path)
    if audio_data.ndim != 2 or audio_data.shape[1] < 2:
        raise RuntimeError(
            f'Finger audio must be stereo (L=piezo, R=air); got shape {audio_data.shape} in {audio_path}'
        )

    finger_piezo = audio_data[:, 0]
    finger_air = audio_data[:, 1]
    finger_grp.create_array('finger_piezo', data=finger_piezo, compressor=_BLOSC)
    finger_grp.create_array('finger_air', data=finger_air, compressor=_BLOSC)
    audio_ts = _audio_timestamps(meta.audio_start_time_ns, len(finger_piezo), sr)
    ts_grp.create_array('finger_piezo', data=audio_ts, compressor=_BLOSC)
    ts_grp.create_array('finger_air', data=audio_ts, compressor=_BLOSC)

    # --- GoPro ---
    if skip_gopro:
        log.info('  Skipping GoPro frames (--skip-gopro).')
    else:
        gopro_path = session.path / GOPRO_MP4
        if not gopro_path.exists():
            log.warning(f'  No gopro.mp4 found at {gopro_path}, skipping GoPro frames.')
        else:
            _write_gopro_frames(ep_grp, gopro_path)

    # --- Annotations ---
    ann_attrs: dict[str, float] = {
        'episode_start': float(finger_ts[0]),
        'episode_end': float(finger_ts[-1]),
    }
    if meta.sync_chirp_play_time_ns is not None:
        ann_attrs['sync_chirp_play_time_s'] = meta.sync_chirp_play_time_ns / 1e9
    ann_grp.attrs.update(ann_attrs)


def build_pzarr(scene_path: pathlib.Path, skip_gopro: bool = False) -> pathlib.Path:
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
    optitrack_start_time = first_meta.optitrack_start_time
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
            'optitrack_start_time': (optitrack_start_time.isoformat() if optitrack_start_time is not None else None),
        }
    )

    if optitrack_start_time is not None:
        csv_path = find_optitrack_csv(scene_path)
        if csv_path is not None:
            log.info(f'OptiTrack CSV found: {csv_path.name}')
            write_optitrack(root, csv_path, optitrack_start_time.timestamp())
        else:
            log.warning(
                f'optitrack_start_time is set in {scene_path.name} but no OptiTrack CSV'
                f' (Format Version,1.23) was found in that scene directory.'
                f' OptiTrack poses will not be included in the zarr store.'
            )

    for i, session in enumerate(sessions):
        log.info(f'[{i + 1}/{len(sessions)}] Episode {i}: {session.path.name}')
        ep_grp = root.require_group(f'episode_{i}')
        ep_grp.attrs['session_type'] = session.metadata.session_type.value
        ep_grp.attrs['session_dir'] = session.path.name
        _write_episode(ep_grp, session, skip_gopro)

    return scene.zarr_path


def _ts_freq_hz(ts: np.ndarray) -> float | None:
    """Return median sample rate in Hz from a 1-D timestamp array (seconds)."""
    if len(ts) < 2:
        return None
    median_dt = float(np.median(np.diff(ts)))
    return 1.0 / median_dt if median_dt > 0 else None


@dataclasses.dataclass
class StreamInfo:
    """Shape, timestamp range, and sample rate for one data stream."""

    shape: tuple | None  # type: ignore[type-arg]
    ts_range: tuple[float, float] | None
    freq_hz: float | None


@dataclasses.dataclass
class OptitrackInfo:
    """Summary of the scene-level optitrack group."""

    shape: tuple  # type: ignore[type-arg]
    ts_range: tuple[float, float]
    freq_hz: float | None


@dataclasses.dataclass
class EpisodeInfo:
    """Summary of one episode's arrays and timestamps extracted from scene.zarr."""

    index: int
    finger: StreamInfo
    finger_piezo: StreamInfo
    finger_air: StreamInfo
    gopro: StreamInfo
    gopro_accl: StreamInfo
    gopro_gyro: StreamInfo
    gopro_gps: StreamInfo
    gopro_audio: StreamInfo
    episode_start: float | None
    episode_end: float | None

    # Legacy shape/range attributes kept for backward compatibility
    @property
    def finger_shape(self) -> tuple | None:  # type: ignore[type-arg]
        """Shape of finger/frames array."""
        return self.finger.shape

    @property
    def finger_piezo_shape(self) -> tuple | None:  # type: ignore[type-arg]
        """Shape of finger/finger_piezo array."""
        return self.finger_piezo.shape

    @property
    def finger_air_shape(self) -> tuple | None:  # type: ignore[type-arg]
        """Shape of finger/finger_air array."""
        return self.finger_air.shape

    @property
    def gopro_shape(self) -> tuple | None:  # type: ignore[type-arg]
        """Shape of gopro/frames array."""
        return self.gopro.shape

    @property
    def accl_shape(self) -> tuple | None:  # type: ignore[type-arg]
        """Shape of gopro/accl array."""
        return self.gopro_accl.shape

    @property
    def gyro_shape(self) -> tuple | None:  # type: ignore[type-arg]
        """Shape of gopro/gyro array."""
        return self.gopro_gyro.shape

    @property
    def gps_shape(self) -> tuple | None:  # type: ignore[type-arg]
        """Shape of gopro/gps array."""
        return self.gopro_gps.shape

    @property
    def gopro_audio_shape(self) -> tuple | None:  # type: ignore[type-arg]
        """Shape of gopro/audio array."""
        return self.gopro_audio.shape

    @property
    def finger_ts_range(self) -> tuple[float, float] | None:
        """Timestamp range for finger/frames."""
        return self.finger.ts_range

    @property
    def finger_ts_mean_delta_ms(self) -> float | None:
        """Mean inter-frame delta in ms for finger camera."""
        if self.finger.freq_hz and self.finger.freq_hz > 0:
            return 1000.0 / self.finger.freq_hz
        return None

    @property
    def finger_piezo_ts_range(self) -> tuple[float, float] | None:
        """Timestamp range for finger/finger_piezo."""
        return self.finger_piezo.ts_range

    @property
    def finger_air_ts_range(self) -> tuple[float, float] | None:
        """Timestamp range for finger/finger_air."""
        return self.finger_air.ts_range

    @property
    def gopro_ts_range(self) -> tuple[float, float] | None:
        """Timestamp range for gopro/frames."""
        return self.gopro.ts_range

    @property
    def gopro_ts_mean_delta_ms(self) -> float | None:
        """Mean inter-frame delta in ms for GoPro."""
        if self.gopro.freq_hz and self.gopro.freq_hz > 0:
            return 1000.0 / self.gopro.freq_hz
        return None

    @property
    def gopro_audio_ts_range(self) -> tuple[float, float] | None:
        """Timestamp range for gopro/audio."""
        return self.gopro_audio.ts_range


@dataclasses.dataclass
class PZarrInfo:
    """Top-level summary of a scene.zarr store returned by inspect_pzarr."""

    zarr_path: pathlib.Path
    zarr_format: int
    tree: object
    attrs: dict  # type: ignore[type-arg]
    episodes: list[EpisodeInfo]
    optitrack: OptitrackInfo | None


def _read_stream_info(ep: zarr.Group, array_path: str, ts_path: str) -> StreamInfo:
    """Read shape and timestamp stats for one stream from an episode group."""
    shape = arr(ep, array_path).shape if array_path in ep else None
    ts_range: tuple[float, float] | None = None
    freq_hz: float | None = None
    if ts_path in ep:
        ts: np.ndarray = arr(ep, ts_path)[:]  # type: ignore[assignment]
        ts_range = (float(ts[0]), float(ts[-1]))
        freq_hz = _ts_freq_hz(ts)
    return StreamInfo(shape=shape, ts_range=ts_range, freq_hz=freq_hz)


def inspect_pzarr(scene_path: pathlib.Path) -> PZarrInfo:
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

        ann_attrs = ep['annotations'].attrs if 'annotations' in ep else {}
        ep_start = float(ann_attrs['episode_start']) if 'episode_start' in ann_attrs else None
        ep_end = float(ann_attrs['episode_end']) if 'episode_end' in ann_attrs else None
        episodes.append(
            EpisodeInfo(
                index=i,
                finger=_read_stream_info(ep, 'finger/frames', 'timestamps/finger'),
                finger_piezo=_read_stream_info(ep, 'finger/finger_piezo', 'timestamps/finger_piezo'),
                finger_air=_read_stream_info(ep, 'finger/finger_air', 'timestamps/finger_air'),
                gopro=_read_stream_info(ep, 'gopro/frames', 'timestamps/gopro'),
                gopro_accl=_read_stream_info(ep, 'gopro/accl', 'timestamps/gopro_accl'),
                gopro_gyro=_read_stream_info(ep, 'gopro/gyro', 'timestamps/gopro_gyro'),
                gopro_gps=_read_stream_info(ep, 'gopro/gps', 'timestamps/gopro_gps'),
                gopro_audio=_read_stream_info(ep, 'gopro/audio', 'timestamps/gopro_audio'),
                episode_start=ep_start,
                episode_end=ep_end,
            )
        )

    optitrack: OptitrackInfo | None = None
    if 'optitrack' in root:
        ot = zarr.open_group(str(zarr_path / 'optitrack'), mode='r')
        if 'timestamps' in ot and 'pose' in ot:
            ot_ts: np.ndarray = arr(ot, 'timestamps')[:]  # type: ignore[assignment]
            optitrack = OptitrackInfo(
                shape=arr(ot, 'pose').shape,
                ts_range=(float(ot_ts[0]), float(ot_ts[-1])),
                freq_hz=_ts_freq_hz(ot_ts),
            )

    return PZarrInfo(
        zarr_path=zarr_path,
        zarr_format=root.metadata.zarr_format,
        tree=root.tree(),
        attrs=dict(root.attrs),
        episodes=episodes,
        optitrack=optitrack,
    )


def read_frame(scene_path: pathlib.Path, episode: int = 0, frame: int = 0) -> np.ndarray:
    """Read a single frame from scene.zarr; returns (H, W, 3) uint8 RGB array."""
    zarr_path = SceneFiles.resolve_zarr_path(scene_path)
    root = zarr.open_group(str(zarr_path), mode='r')
    return arr(root, f'episode_{episode}/finger/frames')[frame]  # type: ignore[return-value]
