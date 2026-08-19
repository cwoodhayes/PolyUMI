"""Tests for downstream-step invalidation when an earlier step is re-run."""

import pathlib
import unittest.mock as mock

import pytest
import zarr

from polyumi_ingest.preproc.step_base import (
    _set_steps_needing_rebuild,
    run_preprocessing,
    steps_needing_rebuild,
    _invalidate_downstream_steps,
    _mark_preprocessing_step,
    episode_steps_done,
    preprocessing_step_versions,
    preprocessing_steps_done,
)


def _scene(tmp_path: pathlib.Path, steps: list[int], n_episodes: int = 2) -> zarr.Group:
    """Build a store where every step in ``steps`` is complete, at scene and episode level."""
    root = zarr.open_group(str(tmp_path / 'scene.zarr'), mode='w', zarr_format=2)
    root.attrs['preprocessing_steps'] = sorted(steps)
    root.attrs['preprocessing_step_versions'] = {
        str(s): {'git_sha': f'sha{s}', 'completed_at': f'2026-01-0{s}T00:00:00+00:00'} for s in steps
    }
    for i in range(n_episodes):
        ep = root.require_group(f'episode_{i}')
        ep.attrs['preprocessing_steps'] = sorted(steps)
    return root


def test_rerunning_a_step_marks_the_later_ones_incomplete(tmp_path: pathlib.Path) -> None:
    """
    The bug this exists for: step 2 re-ran and steps 3-5 stayed 'done' against stale inputs.

    Scene 31d6 exported 18 episodes of ``eef/pose_slam`` built from a superseded SLAM pass
    because nothing invalidated step 5 when step 2 was re-run 90 minutes later.
    """
    root = _scene(tmp_path, [1, 2, 3, 4, 5])

    _mark_preprocessing_step(root, 2)

    assert preprocessing_steps_done(root) == [1, 2]
    assert sorted(preprocessing_step_versions(root)) == ['1', '2']


def test_invalidation_reaches_the_per_episode_marks(tmp_path: pathlib.Path) -> None:
    """
    Scene-level marks alone are not enough to make a step actually re-run.

    ``run_step``'s episode loop skips any episode already carrying the step's number, so
    leaving those behind would re-open the step and then skip every episode in it.
    """
    root = _scene(tmp_path, [1, 2, 3, 4, 5])

    _mark_preprocessing_step(root, 2)

    for key in ('episode_0', 'episode_1'):
        assert episode_steps_done(root[key]) == [1, 2]


def test_marking_the_last_step_invalidates_nothing(tmp_path: pathlib.Path) -> None:
    """The common case — finishing the pipeline in order — must clear nothing."""
    root = _scene(tmp_path, [1, 2, 3, 4])

    _mark_preprocessing_step(root, 5)

    assert preprocessing_steps_done(root) == [1, 2, 3, 4, 5]
    assert episode_steps_done(root['episode_0']) == [1, 2, 3, 4]


def test_invalidation_leaves_the_arrays_alone(tmp_path: pathlib.Path) -> None:
    """Only completion marks are dropped; the stale step's output waits to be overwritten."""
    root = _scene(tmp_path, [1, 2, 5])
    root.require_group('episode_0').require_group('eef').create_array('pose_slam', shape=(3, 7), dtype='f8')

    _invalidate_downstream_steps(root, 2)

    assert 'eef/pose_slam' in root['episode_0']


def test_invalidated_steps_are_recorded_for_rebuild(tmp_path: pathlib.Path) -> None:
    """
    Clearing marks is not enough — the debt has to persist for the run that rebuilds them.

    Invalidation and rebuild are routinely different processes: `pingest pp 2 --force`
    today, `pingest pp` tomorrow.
    """
    root = _scene(tmp_path, [1, 2, 3, 4, 5])

    _mark_preprocessing_step(root, 2)

    assert steps_needing_rebuild(root) == {3, 4, 5}


def test_rebuild_debt_survives_reopening_the_store(tmp_path: pathlib.Path) -> None:
    """It lives on the store, not in memory, so a later invocation still honours it."""
    root = _scene(tmp_path, [1, 2, 3, 4, 5])
    _mark_preprocessing_step(root, 2)

    reopened = zarr.open_group(str(tmp_path / 'scene.zarr'), mode='a')

    assert steps_needing_rebuild(reopened) == {3, 4, 5}


def test_a_clean_run_records_no_rebuild_debt(tmp_path: pathlib.Path) -> None:
    """Finishing the pipeline in order must not leave anything marked for recompute."""
    root = _scene(tmp_path, [1, 2, 3, 4])

    _mark_preprocessing_step(root, 5)

    assert steps_needing_rebuild(root) == set()


def test_run_preprocessing_forces_an_invalidated_step(tmp_path: pathlib.Path) -> None:
    """
    The bug behind the bug: an invalidated step was re-entered but declined to recompute.

    so_align and eef-pose both carry an "output already present; use --force to recompute"
    guard, so rebuilding scene 31d6 logged "eef/pose_* already present" for all 63 episodes
    and kept the stale poses through a run that looked successful from the outside.
    """
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    root.attrs['preprocessing_steps'] = [1, 2]
    root.attrs['n_episodes'] = 1
    root.require_group('episode_0').attrs['preprocessing_steps'] = [1, 2]
    _set_steps_needing_rebuild(root, {2})

    seen: dict[int, bool] = {}

    class _FakeStep:
        step_name = 'fake'
        step_number = 2

        def run(self, path, copy=False, force=False):
            seen[self.step_number] = force
            return path

    with mock.patch.dict(
        'polyumi_ingest.preproc.step_base.PREPROCESSING_STEPS',
        {2: _FakeStep},
        clear=True,
    ):
        run_preprocessing(scene_zarr, step_number=2)

    assert seen[2] is True, 'an invalidated step must run as if forced'
    # and the debt is cleared once it has actually been rebuilt
    assert steps_needing_rebuild(zarr.open_group(str(scene_zarr), mode='r')) == set()


def test_a_forced_step_still_invalidates_the_ones_after_it(tmp_path: pathlib.Path) -> None:
    """
    Clearing marks for a forced run must not swallow the downstream debt it then creates.

    `pingest pp 2 --force` drops step 2's own marks up front (so an interrupted run can't
    leave them falsely complete), then re-runs it — which makes steps 3+ describe superseded
    inputs, exactly as an unforced re-run would. The two mechanisms touch the same attrs, so
    this pins that they compose: the forced step's marks go, its successors' debt appears.
    """
    root = _scene(tmp_path, steps=[1, 2, 3])
    scene_zarr = tmp_path / 'scene.zarr'

    class _FakeStep:
        step_name = 'fake'
        step_number = 2

        def run(self, path, copy=False, force=False):
            return path

    with mock.patch.dict(
        'polyumi_ingest.preproc.step_base.PREPROCESSING_STEPS',
        {2: _FakeStep},
        clear=True,
    ):
        run_preprocessing(scene_zarr, step_number=2, force=True)

    root = zarr.open_group(str(scene_zarr), mode='r')
    assert preprocessing_steps_done(root) == [1, 2]  # 3 invalidated by the re-run
    assert steps_needing_rebuild(root) == {3}
    # 2 cleared by the force, 3 by the invalidation. The stand-in step does no per-episode
    # work, so nothing puts 2 back here — the real one re-marks each episode as it finishes.
    assert episode_steps_done(root['episode_0']) == [1]


def test_a_targeted_step_refuses_to_build_on_unrebuilt_debt(tmp_path: pathlib.Path) -> None:
    """
    `pingest pp 5` must not quietly rebuild step 5 from step 3/4 output known to be stale.

    The debt list is the only record left of it — invalidation pops those steps' ``completed_at``
    stamps, so ``_stale_steps`` cannot see them either.
    """
    root = _scene(tmp_path, steps=[1, 2])
    _set_steps_needing_rebuild(root, {3, 4, 5})
    scene_zarr = tmp_path / 'scene.zarr'

    ran: list[int] = []

    class _FakeStep:
        step_name = 'fake'
        step_number = 0

        def run(self, path, copy=False, force=False):
            ran.append(self.step_number)
            return path

    steps = {n: type(f'_Fake{n}', (_FakeStep,), {'step_number': n}) for n in (3, 4, 5)}
    with mock.patch.dict('polyumi_ingest.preproc.step_base.PREPROCESSING_STEPS', steps, clear=True):
        with pytest.raises(RuntimeError, match=r'built from step\(s\) \[3, 4\]'):
            run_preprocessing(scene_zarr, step_number=5)
        assert ran == []

        # The full run is legal: 3 and 4 are rebuilt by the same loop, before 5 needs them.
        run_preprocessing(scene_zarr, step_number=None)

    assert ran == [3, 4, 5]
