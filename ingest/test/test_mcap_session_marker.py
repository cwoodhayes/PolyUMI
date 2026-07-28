"""Tests for the /session/events chirp-end marker exported to MCAP for Foxglove."""

import json
import pathlib

import numpy as np
import pytest
import zarr
from mcap.reader import make_reader
from polyumi_pi import sync_chirp

from polyumi_ingest.export.mcap import export_episode_to_mcap


def _build_episode(tmp_path: pathlib.Path, *, finger_chirp_onset_s: float | None) -> tuple[zarr.Group, zarr.Group]:
    """Build a minimal 2-frame episode, optionally with a chirp-time-sync annotation."""
    n = 2
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')

    finger_frames = np.zeros((n, 8, 8, 3), dtype=np.uint8)
    finger_audio = np.zeros(64, dtype=np.float32)
    finger_ts = np.arange(n, dtype=np.float64) / 10.0
    audio_ts = np.arange(64, dtype=np.float64) / 16_000.0
    ep.create_group('finger').create_array('frames', data=finger_frames)
    ep['finger'].create_array('finger_piezo', data=finger_audio)  # type: ignore[union-attr]
    ep['finger'].create_array('finger_air', data=finger_audio)  # type: ignore[union-attr]
    ts_grp = ep.create_group('timestamps')
    ts_grp.create_array('finger', data=finger_ts)
    ts_grp.create_array('finger_piezo', data=audio_ts)
    ts_grp.create_array('finger_air', data=audio_ts)

    if finger_chirp_onset_s is not None:
        ts_sync = ep.create_group('annotations').create_group('time_sync')
        ts_sync.attrs['finger_chirp_onset_s'] = finger_chirp_onset_s

    return root, ep


def _session_events(mcap_path: pathlib.Path) -> list[dict]:
    with mcap_path.open('rb') as f:
        reader = make_reader(f)
        return [json.loads(msg.data) for _, _, msg in reader.iter_messages(topics=['/session/events'])]


def test_chirp_marker_written_at_onset_plus_duration(tmp_path: pathlib.Path) -> None:
    """The marker fires at finger_chirp_onset_s + DURATION_S, on the finger (MCAP) clock."""
    onset_s = 3.25
    root, ep = _build_episode(tmp_path, finger_chirp_onset_s=onset_s)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    events = _session_events(mcap_path)
    assert len(events) == 1
    expected_end_s = onset_s + sync_chirp.DURATION_S
    got_s = events[0]['timestamp']['sec'] + events[0]['timestamp']['nsec'] / 1e9
    assert got_s == pytest.approx(expected_end_s, abs=1e-6)
    assert 'episode start' in events[0]['message'].lower()


def test_no_session_events_topic_without_chirp_annotation(tmp_path: pathlib.Path) -> None:
    """No annotations/time_sync at all (step 1 never ran) — no /session/events messages."""
    root, ep = _build_episode(tmp_path, finger_chirp_onset_s=None)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    assert _session_events(mcap_path) == []
