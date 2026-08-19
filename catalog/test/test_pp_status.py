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


def test_scene_pp_status_ignores_unregistered_step_numbers(tmp_path: pathlib.Path):
    """
    A stale/unregistered step number in the attr doesn't inflate n_complete past n_total.

    Regression test: n_complete used to be len(completed) directly, so a scene processed
    under a since-retired step number (e.g. 999, never registered) would report more steps
    complete than actually exist.
    """
    from polyumi_ingest.preproc import available_preprocessing_steps

    scene_dir = tmp_path / 'scene_stale'
    scene_dir.mkdir()
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [1, 999]

    status = pp_status.scene_pp_status(scene_dir)

    assert status['n_total'] == len(available_preprocessing_steps())
    assert status['n_complete'] == 1
    assert status['n_complete'] <= status['n_total']


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

    monkeypatch.setattr('polyumi_ingest.pzarr.store.build_pzarr', fake_build_pzarr)
    monkeypatch.setattr('polyumi_ingest.preproc.run_preprocessing', fake_run_preprocessing)

    pp_status.run_full_pipeline(scene_dir)

    assert [c[0] for c in calls] == ['build_pzarr', 'run_preprocessing']


def test_run_full_pipeline_skips_build_when_pzarr_exists(tmp_path: pathlib.Path, monkeypatch):
    """When scene.zarr already exists, only run_preprocessing is called."""
    scene_dir = tmp_path / 'scene_f'
    scene_dir.mkdir()
    zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')

    calls = []
    monkeypatch.setattr('polyumi_ingest.pzarr.store.build_pzarr', lambda *a, **k: calls.append('build_pzarr'))
    monkeypatch.setattr('polyumi_ingest.preproc.run_preprocessing', lambda *a, **k: calls.append('run_preprocessing'))

    pp_status.run_full_pipeline(scene_dir)

    assert calls == ['run_preprocessing']


def test_run_full_pipeline_appends_sessions_recorded_since_the_build(tmp_path: pathlib.Path, monkeypatch):
    """
    A scene that grew after its store was built gets extended, not silently left behind.

    The Fetch button pulls new sessions per session, so the store is routinely behind the
    directory by the time Run pp is pressed. Skipping the build whenever scene.zarr merely
    *exists* would preprocess only the old episodes and never say so.
    """
    scene_dir = tmp_path / 'scene_grown'
    scene_dir.mkdir()
    _make_session(scene_dir, 'session_1', with_gopro=True)
    _make_session(scene_dir, 'session_2', with_gopro=True)
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.require_group('episode_0').attrs['session_dir'] = 'session_1'  # session_2 is new

    kwargs_seen = []
    monkeypatch.setattr(
        'polyumi_ingest.pzarr.store.build_pzarr',
        lambda path, **kwargs: kwargs_seen.append(kwargs) or (path / 'scene.zarr'),
    )
    monkeypatch.setattr('polyumi_ingest.preproc.run_preprocessing', lambda *a, **k: None)

    pp_status.run_full_pipeline(scene_dir)

    assert kwargs_seen == [{'skip_gopro': False, 'append': True}]


def test_run_full_pipeline_passes_force_through_to_run_preprocessing(tmp_path: pathlib.Path, monkeypatch):
    """force=True must reach run_preprocessing, not get silently dropped along the way."""
    scene_dir = tmp_path / 'scene_g'
    scene_dir.mkdir()
    zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')

    forces_seen = []
    monkeypatch.setattr(
        'polyumi_ingest.preproc.run_preprocessing',
        lambda *a, force=False, **k: forces_seen.append(force),
    )

    pp_status.run_full_pipeline(scene_dir, force=True)
    pp_status.run_full_pipeline(scene_dir)  # default

    assert forces_seen == [True, False]


def test_scene_pp_status_exposes_per_step_commit(tmp_path: pathlib.Path):
    """Each completed step carries the git sha + timestamp recorded when it ran."""
    scene_dir = tmp_path / 'scene_prov'
    scene_dir.mkdir()
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [1, 2]
    root.attrs['preprocessing_step_versions'] = {
        '1': {'git_sha': 'a' * 40, 'completed_at': '2026-08-01T00:00:00+00:00'},
        '2': {'git_sha': 'b' * 40, 'completed_at': '2026-08-01T01:00:00+00:00'},
    }

    steps = {s['number']: s for s in pp_status.scene_pp_status(scene_dir)['steps']}

    assert steps[1]['git_sha'] == 'a' * 40
    assert steps[2]['completed_at'] == '2026-08-01T01:00:00+00:00'


def test_scene_pp_status_reports_none_sha_for_stores_predating_provenance(tmp_path: pathlib.Path):
    """
    A store processed before per-step provenance existed reports complete steps with no sha.

    The completion mark stays authoritative — a missing version entry must not be read as
    "step didn't run", which would silently un-tick every already-processed scene.
    """
    scene_dir = tmp_path / 'scene_legacy'
    scene_dir.mkdir()
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [1]

    steps = {s['number']: s for s in pp_status.scene_pp_status(scene_dir)['steps']}

    assert steps[1]['complete'] is True
    assert steps[1]['git_sha'] is None
    assert steps[1]['completed_at'] is None


def test_reset_pp_status_clears_completion_marks(tmp_path: pathlib.Path):
    """Resetting drops every step back to incomplete and clears the recorded commits."""
    scene_dir = tmp_path / 'scene_reset'
    scene_dir.mkdir()
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [1, 2, 3]
    root.attrs['preprocessing_step_versions'] = {'1': {'git_sha': 'a' * 40, 'completed_at': 'x'}}

    pp_status.reset_pp_status(scene_dir)

    status = pp_status.scene_pp_status(scene_dir)
    assert status['n_complete'] == 0
    assert all(not s['complete'] for s in status['steps'])
    assert all(s['git_sha'] is None for s in status['steps'])


def test_reset_pp_status_is_a_noop_without_pzarr(tmp_path: pathlib.Path):
    """Nothing to reset when the store doesn't exist yet, and no store gets created."""
    scene_dir = tmp_path / 'scene_nopzarr'
    scene_dir.mkdir()

    pp_status.reset_pp_status(scene_dir)

    assert not (scene_dir / 'scene.zarr').exists()


def test_run_full_pipeline_delegates_the_force_reset_to_ingest(tmp_path: pathlib.Path, monkeypatch):
    """
    Forcing clears the marks inside run_preprocessing, not here.

    It has to reach the *per-episode* marks as well as the root's, and those gate ingest's own
    episode loop — clearing only the root's here left a forced run that died part-way able to
    skip every episode on the next run and mark the step complete anyway. The route still
    clears up front for its own rendering (see test_app), which is why nothing is asserted
    about the marks at this point.
    """
    scene_dir = tmp_path / 'scene_force_reset'
    scene_dir.mkdir()
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [1, 2, 3]

    calls = []
    monkeypatch.setattr('polyumi_ingest.preproc.run_preprocessing', lambda *a, **k: calls.append(k))

    pp_status.run_full_pipeline(scene_dir, force=True)

    assert [c['force'] for c in calls] == [True]


def test_reset_pp_status_also_clears_per_episode_marks(tmp_path: pathlib.Path):
    """
    The route's up-front reset must reach both levels, or it re-creates the bug it hides.

    Root-only clearing leaves each episode still claiming the step, so the run that follows
    skips every episode and re-stamps the scene complete having computed nothing.
    """
    scene_dir = tmp_path / 'scene_reset_episodes'
    scene_dir.mkdir()
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 1
    root.attrs['preprocessing_steps'] = [1, 2]
    root.create_group('episode_0').attrs['preprocessing_steps'] = [1, 2]

    pp_status.reset_pp_status(scene_dir)

    reopened = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='r')
    assert list(reopened.attrs['preprocessing_steps']) == []
    assert list(reopened['episode_0'].attrs['preprocessing_steps']) == []


def test_run_full_pipeline_without_force_keeps_completion_marks(tmp_path: pathlib.Path, monkeypatch):
    """
    The non-forced "continue" run must not reset anything.

    Its whole point is to skip already-complete steps; clearing the marks would make it
    re-run the entire pipeline.
    """
    scene_dir = tmp_path / 'scene_continue'
    scene_dir.mkdir()
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = 0
    root.attrs['preprocessing_steps'] = [1, 2]

    monkeypatch.setattr('polyumi_ingest.preproc.run_preprocessing', lambda *a, **k: None)

    pp_status.run_full_pipeline(scene_dir, force=False)

    assert pp_status.scene_pp_status(scene_dir)['n_complete'] == 2
