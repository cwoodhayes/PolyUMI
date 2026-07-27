"""Tests for the scene-level pp status + full-pipeline trigger (Phase 4)."""

from __future__ import annotations

import pathlib

import zarr
from polyumi_catalog import pp_status
from polyumi_pi.files.metadata import SessionMetadata, SessionType


def _make_session(scene_dir: pathlib.Path, name: str, *, with_gopro: bool) -> pathlib.Path:
    sd = scene_dir / name
    sd.mkdir(parents=True)
    SessionMetadata(
        path=sd / 'metadata.json',
        scene_id='scene-1',
        session_type=SessionType.EPISODE,
        n_video_frames=10,
    ).to_file()
    if with_gopro:
        (sd / 'gopro.mp4').write_bytes(b'fake-mp4')
    return sd


def test_scene_pp_status_without_pzarr_reports_no_steps_complete(tmp_path: pathlib.Path):
    """No scene.zarr at all reports pzarr_exists=False and every step incomplete."""
    scene_dir = tmp_path / 'scene_a'
    scene_dir.mkdir()

    status = pp_status.scene_pp_status(scene_dir)

    assert status['pzarr_exists'] is False
    assert status['n_complete'] == 0
    assert status['n_total'] > 0
    assert all(not s['complete'] for s in status['steps'])


def test_scene_pp_status_reflects_completed_steps(tmp_path: pathlib.Path):
    """Steps recorded in the scene.zarr's preprocessing_steps attr show as complete."""
    scene_dir = tmp_path / 'scene_b'
    scene_dir.mkdir()
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [1, 2]

    status = pp_status.scene_pp_status(scene_dir)

    assert status['pzarr_exists'] is True
    assert status['n_complete'] == 2
    complete_numbers = {s['number'] for s in status['steps'] if s['complete']}
    assert complete_numbers == {1, 2}


def test_scene_pp_status_does_not_call_inspect_pzarr(tmp_path: pathlib.Path, monkeypatch):
    """
    Regression test: scene_pp_status must not go through inspect_pzarr.

    inspect_pzarr reads every episode's full per-sample timestamp arrays to compute
    stream shapes/rates that scene_pp_status doesn't use; since this is called on every
    scene selection, it needs to stay a cheap, attrs-only read.
    """

    def _boom(*args, **kwargs):
        raise AssertionError('scene_pp_status must not call inspect_pzarr')

    monkeypatch.setattr('polyumi_ingest.pzarr.inspect_pzarr', _boom)

    scene_dir = tmp_path / 'scene_g'
    scene_dir.mkdir()
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 1
    root.attrs['preprocessing_steps'] = [1]

    status = pp_status.scene_pp_status(scene_dir)
    assert status['n_complete'] == 1


def test_missing_gopro_mp4s_lists_only_sessions_without_it(tmp_path: pathlib.Path):
    """missing_gopro_mp4s reports session dirs lacking gopro.mp4, and only those."""
    scene_dir = tmp_path / 'scene_c'
    scene_dir.mkdir()
    _make_session(scene_dir, 'session_1', with_gopro=True)
    _make_session(scene_dir, 'session_2', with_gopro=False)

    missing = pp_status.missing_gopro_mp4s(scene_dir)

    assert missing == ['session_2']


def test_run_full_pipeline_raises_when_gopro_missing(tmp_path: pathlib.Path):
    """Without scene.zarr and a session missing gopro.mp4, it raises rather than building silently."""
    scene_dir = tmp_path / 'scene_d'
    scene_dir.mkdir()
    _make_session(scene_dir, 'session_1', with_gopro=False)

    try:
        pp_status.run_full_pipeline(scene_dir)
    except FileNotFoundError as exc:
        assert 'gopro.mp4' in str(exc)
        assert 'session_1' in str(exc)
    else:
        raise AssertionError('expected FileNotFoundError')


def test_run_full_pipeline_builds_pzarr_then_runs_preprocessing(tmp_path: pathlib.Path, monkeypatch):
    """When scene.zarr is missing, pzarr is built first, then run_preprocessing runs on it."""
    scene_dir = tmp_path / 'scene_e'
    scene_dir.mkdir()
    _make_session(scene_dir, 'session_1', with_gopro=True)

    calls = []

    def fake_build_pzarr(path, **kwargs):
        calls.append(('build_pzarr', path))
        zarr.open_group(str(path / 'scene.zarr'), mode='w')
        return path / 'scene.zarr'

    def fake_run_preprocessing(path, **kwargs):
        calls.append(('run_preprocessing', path))
        return path / 'scene.zarr'

    monkeypatch.setattr('polyumi_ingest.pzarr.build_pzarr', fake_build_pzarr)
    monkeypatch.setattr('polyumi_ingest.preproc.run_preprocessing', fake_run_preprocessing)

    pp_status.run_full_pipeline(scene_dir)

    assert [c[0] for c in calls] == ['build_pzarr', 'run_preprocessing']


def test_run_full_pipeline_skips_build_when_pzarr_exists(tmp_path: pathlib.Path, monkeypatch):
    """When scene.zarr already exists, only run_preprocessing is called."""
    scene_dir = tmp_path / 'scene_f'
    scene_dir.mkdir()
    zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')

    calls = []
    monkeypatch.setattr('polyumi_ingest.pzarr.build_pzarr', lambda *a, **k: calls.append('build_pzarr'))
    monkeypatch.setattr('polyumi_ingest.preproc.run_preprocessing', lambda *a, **k: calls.append('run_preprocessing'))

    pp_status.run_full_pipeline(scene_dir)

    assert calls == ['run_preprocessing']
