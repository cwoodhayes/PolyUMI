"""
End-to-end fault isolation in ``build_pzarr``.

This is the reported failure reproduced at scene level: one session with a zero-byte final
JPEG, one whose ``audio.wav`` is empty enough that it never loads at all, and healthy sessions
either side of them.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import wave

import cv2
import numpy as np
import zarr
from polyumi_ingest.episode_status import Episode
from polyumi_ingest.manifests import SceneManifest
from polyumi_ingest.pzarr.store import build_pzarr

_H, _W = 4, 6
_SAMPLE_RATE = 16_000
_SCENE_ID = 'scene-uuid-1234'


def _write_session(
    scene_dir: pathlib.Path,
    name: str,
    n_frames: int = 4,
    empty_frames: set[int] = frozenset(),
    empty_audio: bool = False,
    audio_channels: int = 2,
    created_at: str | None = None,
    session_type: str = 'EPISODE',
) -> pathlib.Path:
    """Write a minimal but genuine session directory: metadata, JPEGs + timestamps, stereo WAV."""
    session_dir = scene_dir / name
    video_dir = session_dir / 'video'
    video_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_frames):
        path = video_dir / f'frame_{i:06d}.jpg'
        if i in empty_frames:
            path.write_bytes(b'')
        else:
            path.write_bytes(cv2.imencode('.jpg', rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8))[1].tobytes())
        rows.append(f'{i},{1_598_685_227_000 + i * 100_000_000}')
    (video_dir / 'video_timestamps.csv').write_text('\n'.join(rows) + '\n')

    audio_path = session_dir / 'audio.wav'
    if empty_audio:
        audio_path.write_bytes(b'')
    else:
        with wave.open(str(audio_path), 'wb') as wf:
            wf.setnchannels(audio_channels)
            wf.setsampwidth(2)
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(rng.integers(-2000, 2000, _SAMPLE_RATE, dtype=np.int16).tobytes())

    (session_dir / 'metadata.json').write_text(
        json.dumps(
            {
                'session_id': f'{name}-id',
                'scene_id': _SCENE_ID,
                'created_at': created_at or f'2026-07-29T20:{30 + len(name) % 20:02d}:19.488374+00:00',
                'duration_s': 1.0,
                'pi_hostname': 'testpi',
                'camera_fps': 10,
                'camera_resolution': [_W, _H],
                'audio_start_time_ns': 1_785_357_020_211_745_762,
                'audio_sample_rate': _SAMPLE_RATE,
                'audio_channels': 2,
                'audio_chunk_ms': 20,
                'n_video_frames': n_frames,
                'n_audio_chunks': 50,
                'video_dropped_frames': 0,
                'audio_dropped_chunks': 0,
                'led_brightness': 1.0,
                'gopro_sync_time': None,
                'first_frame_metadata': {'FrameWallClock': 1_785_357_020_956_362_752},
                'sync_chirp_play_time_ns': None,
                'optitrack_start_time': None,
                'notes': None,
                'task': 'red trapezoid in black mug',
                'robot': 'polyumi_gripper',
                'session_type': session_type,
                'polyumi_version': 'test',
                'file_version': 1,
            }
        )
    )
    return session_dir


def _episode_by_dir(scene_dir: pathlib.Path, session_name: str) -> Episode:
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r')
    for key in root:
        ep = Episode.from_key(zarr.open_group(str(scene_dir / 'scene.zarr'), mode='a'), key)
        if ep.session_dir == session_name:
            return ep
    raise AssertionError(f'No episode built from {session_name}')


def test_truncated_final_frame_does_not_stop_the_scene(tmp_path: pathlib.Path) -> None:
    """A 0-byte last JPEG truncates that one episode; every other episode still builds."""
    scene_dir = tmp_path / 'scene_2026-07-29_16-01-53_2bd6'
    scene_dir.mkdir()
    _write_session(scene_dir, 'session_a')
    _write_session(scene_dir, 'session_b', n_frames=4, empty_frames={3})
    _write_session(scene_dir, 'session_c')

    build_pzarr(scene_dir, skip_gopro=True)

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r')
    assert root.attrs['build_complete'] is True
    assert root.attrs['n_episodes'] == 3

    truncated = _episode_by_dir(scene_dir, 'session_b')
    # One trailing frame lost is a truncation, not a broken episode: it stays usable.
    assert truncated.failure is None
    assert truncated.group['finger/frames'].shape[0] == 3
    assert truncated.group['timestamps/finger'].shape[0] == 3

    for name in ('session_a', 'session_c'):
        healthy = _episode_by_dir(scene_dir, name)
        assert healthy.failure is None
        assert healthy.group['finger/frames'].shape[0] == 4

    # Nothing was flagged, so no manifest needed writing.
    assert SceneManifest.from_scene_dir(scene_dir) is None


def test_broken_episode_is_flagged_and_the_rest_still_build(tmp_path: pathlib.Path) -> None:
    """An episode that fails mid-write is flagged unusable and the scene carries on past it."""
    scene_dir = tmp_path / 'scene_2026-07-29_16-01-53_2bd6'
    scene_dir.mkdir()
    _write_session(scene_dir, 'session_a')
    # Mono audio: loads fine, but _write_episode refuses it (L=piezo, R=air is the contract),
    # so this fails *after* the episode group exists — the case the guard is there for.
    _write_session(scene_dir, 'session_b', audio_channels=1)
    _write_session(scene_dir, 'session_c')

    build_pzarr(scene_dir, skip_gopro=True)

    broken = _episode_by_dir(scene_dir, 'session_b')
    assert broken.failure is not None
    assert broken.failure.step == 'build-pzarr'
    assert 'stereo' in broken.failure.error

    assert _episode_by_dir(scene_dir, 'session_c').group['finger/frames'].shape[0] == 4
    assert _episode_by_dir(scene_dir, 'session_c').failure is None
    assert SceneManifest.from_scene_dir(scene_dir).unusable_episodes == ['session_b']


def test_zero_byte_first_frame_reports_a_readable_reason(tmp_path: pathlib.Path) -> None:
    """
    A session whose *first* frame is empty can't even be loaded, and says so in English.

    VideoFile.from_file decodes frame 0 to learn the resolution, so this fails earlier than
    the per-episode guard — it used to surface as a bare OpenCV ``!buf.empty()`` assertion.
    """
    scene_dir = tmp_path / 'scene_2026-07-29_16-01-53_2bd6'
    scene_dir.mkdir()
    _write_session(scene_dir, 'session_a')
    _write_session(scene_dir, 'session_b', empty_frames={0})

    build_pzarr(scene_dir, skip_gopro=True)

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r')
    assert root.attrs['n_episodes'] == 1
    assert SceneManifest.from_scene_dir(scene_dir).unusable_episodes == ['session_b']


def test_unloadable_session_is_marked_in_scene_json(tmp_path: pathlib.Path) -> None:
    """
    A session too damaged to load never becomes an episode, so it's marked directly.

    Without this it would be a warning nobody reads and a scene that silently has fewer
    episodes than it has session directories.
    """
    scene_dir = tmp_path / 'scene_2026-07-29_16-01-53_2bd6'
    scene_dir.mkdir()
    _write_session(scene_dir, 'session_a')
    _write_session(scene_dir, 'session_b', empty_audio=True)

    build_pzarr(scene_dir, skip_gopro=True)

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r')
    assert root.attrs['n_episodes'] == 1

    manifest = SceneManifest.from_scene_dir(scene_dir)
    assert manifest.unusable_episodes == ['session_b']
    # The manifest we created adopts the scene's real identity from session metadata.
    assert manifest.scene_id == _SCENE_ID


def test_rebuild_clears_a_stale_unusable_mark(tmp_path: pathlib.Path) -> None:
    """
    A session that failed once and now builds clean comes back off ``unusable_episodes``.

    The store is opened mode='w', so the previous run's ``failure`` attr is gone before
    ``episode_guard``'s retry-clearing path could see it — without an explicit reset the
    mark would survive every future rebuild and silently drop the episode from exports.
    """
    scene_dir = tmp_path / 'scene_2026-07-29_16-01-53_2bd6'
    scene_dir.mkdir()
    _write_session(scene_dir, 'session_a')
    _write_session(scene_dir, 'session_b', audio_channels=1)  # mono: rejected by _write_episode

    build_pzarr(scene_dir, skip_gopro=True)
    assert SceneManifest.from_scene_dir(scene_dir).unusable_episodes == ['session_b']

    # Re-record session_b correctly (as a re-fetch would) and rebuild.
    shutil.rmtree(scene_dir / 'session_b')
    _write_session(scene_dir, 'session_b')
    build_pzarr(scene_dir, skip_gopro=True)

    assert _episode_by_dir(scene_dir, 'session_b').failure is None
    assert SceneManifest.from_scene_dir(scene_dir).unusable_episodes == []


def test_interrupted_build_is_detectable(tmp_path: pathlib.Path) -> None:
    """build_complete stays False until every episode has been attempted."""
    scene_dir = tmp_path / 'scene_2026-07-29_16-01-53_2bd6'
    scene_dir.mkdir()
    _write_session(scene_dir, 'session_a')

    from polyumi_ingest.pzarr import pzarr_needs_build

    build_pzarr(scene_dir, skip_gopro=True)
    assert pzarr_needs_build(scene_dir) is False

    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='a')
    root.attrs['build_complete'] = False
    assert pzarr_needs_build(scene_dir) is True

    # A store predating the attribute has no value at all, and must not be rebuilt on sight.
    del root.attrs['build_complete']
    assert pzarr_needs_build(scene_dir) is False
