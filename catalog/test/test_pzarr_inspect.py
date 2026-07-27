"""Tests for the per-session pzarr stream summary (Phase 4), wrapping ingest's inspect_pzarr."""

from __future__ import annotations

import pathlib

import numpy as np
import zarr
from polyumi_catalog import pzarr_inspect


def _make_pzarr_with_finger_stream(scene_dir: pathlib.Path, session_dirname: str, *, n_frames: int = 10) -> None:
    """Build a minimal scene.zarr with one episode's finger/frames stream + episode span."""
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 1
    ep = root.require_group('episode_0')
    ep.attrs['session_dir'] = session_dirname

    finger_grp = ep.require_group('finger')
    finger_grp.create_array('frames', data=np.zeros((n_frames, 4, 4, 3), dtype='uint8'))

    ts_grp = ep.require_group('timestamps')
    ts_grp.create_array('finger', data=np.linspace(0.0, 1.0, n_frames))

    ann_grp = ep.require_group('annotations')
    ann_grp.attrs['episode_start'] = 0.0
    ann_grp.attrs['episode_end'] = 1.0


def test_session_pzarr_streams_without_pzarr_returns_none(tmp_path: pathlib.Path):
    """No scene.zarr at all resolves to None, not an error."""
    scene_dir = tmp_path / 'scene_a'
    scene_dir.mkdir()
    assert pzarr_inspect.session_pzarr_streams(scene_dir, 'session_1') is None


def test_session_pzarr_streams_unknown_session_returns_none(tmp_path: pathlib.Path):
    """A session with no matching episode group resolves to None."""
    scene_dir = tmp_path / 'scene_b'
    scene_dir.mkdir()
    _make_pzarr_with_finger_stream(scene_dir, 'session_1')
    assert pzarr_inspect.session_pzarr_streams(scene_dir, 'session_nonexistent') is None


def test_session_pzarr_streams_reports_present_streams_and_span(tmp_path: pathlib.Path):
    """A matching episode returns its stream shapes/rates and episode span/duration."""
    scene_dir = tmp_path / 'scene_c'
    scene_dir.mkdir()
    _make_pzarr_with_finger_stream(scene_dir, 'session_1', n_frames=10)

    result = pzarr_inspect.session_pzarr_streams(scene_dir, 'session_1')
    assert result is not None
    assert result['episode_index'] == 0
    assert result['episode_start'] == 0.0
    assert result['episode_end'] == 1.0
    assert result['duration_s'] == 1.0

    labels = {s['label'] for s in result['streams']}
    assert 'finger/frames' in labels
    finger = next(s for s in result['streams'] if s['label'] == 'finger/frames')
    assert finger['shape'] == (10, 4, 4, 3)
    assert finger['rate'] is not None
    assert finger['timestamps'] is not None


def test_session_pzarr_streams_omits_absent_streams(tmp_path: pathlib.Path):
    """Streams with no array present (e.g. gopro/frames, dropped from pzarr) are omitted."""
    scene_dir = tmp_path / 'scene_d'
    scene_dir.mkdir()
    _make_pzarr_with_finger_stream(scene_dir, 'session_1')

    result = pzarr_inspect.session_pzarr_streams(scene_dir, 'session_1')
    labels = {s['label'] for s in result['streams']}
    assert 'gopro/frames' not in labels
    assert 'gopro/accl' not in labels
