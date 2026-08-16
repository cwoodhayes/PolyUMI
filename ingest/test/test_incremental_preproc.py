"""
Re-running the pipeline on a scene that gained episodes.

``preprocessing_steps`` on the scene root says a step finished, but a scene grows: episodes
recorded after that run have had nothing done to them. These tests pin the two halves of the
answer — the gate notices outstanding episodes, and the episode loop skips the finished ones,
so the expensive steps only pay for what is actually new.
"""

from __future__ import annotations

import pathlib

import pytest
import zarr
from polyumi_ingest.episode_status import Episode, SceneContext
from polyumi_ingest.preproc.step_base import (
    PREPROCESSING_STEPS,
    PreprocessingStep,
    StepComplete,
    run_preprocessing,
)


class _CountingStep(PreprocessingStep):
    """A registered-looking step that records which episodes it was asked to process."""

    step_number = 1
    step_name = 'counting-step'
    processed: list[str] = []
    complete_early = False

    def prepare_scene(self, scene: SceneContext) -> None:
        if type(self).complete_early:
            raise StepComplete('nothing to do here.')

    def process_episode(self, scene: SceneContext, episode: Episode) -> None:
        type(self).processed.append(episode.key)


@pytest.fixture
def only_counting_step(monkeypatch):
    """Swap the real registry for one step, so run_preprocessing is cheap and deterministic."""
    monkeypatch.setattr(
        'polyumi_ingest.preproc.step_base.PREPROCESSING_STEPS',
        {1: _CountingStep},
    )
    _CountingStep.processed = []
    _CountingStep.complete_early = False
    yield _CountingStep
    assert 1 in PREPROCESSING_STEPS  # the real registry is untouched outside the patch


def _scene(tmp_path: pathlib.Path, n_episodes: int, episode_marks: list[int] | None = None) -> pathlib.Path:
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    for i in range(n_episodes):
        ep = root.create_group(f'episode_{i}')
        ep.attrs['session_dir'] = f'session_{i}'
        if episode_marks is not None:
            ep.attrs['preprocessing_steps'] = list(episode_marks)
    return scene_zarr


def test_a_new_episode_reopens_a_completed_step(tmp_path: pathlib.Path, only_counting_step) -> None:
    """The scene says step 1 is done, but the episode appended afterwards still gets it."""
    scene_zarr = _scene(tmp_path, n_episodes=2, episode_marks=[1])
    root = zarr.open_group(str(scene_zarr), mode='a')
    root.attrs['preprocessing_steps'] = [1]
    new = root.create_group('episode_2')
    new.attrs['session_dir'] = 'session_2'
    new.attrs['preprocessing_steps'] = []  # what build_pzarr's append path writes

    run_preprocessing(scene_zarr)

    assert only_counting_step.processed == ['episode_2']
    assert zarr.open_group(str(scene_zarr), mode='r')['episode_2'].attrs['preprocessing_steps'] == [1]


def test_a_fully_processed_scene_stays_skipped(tmp_path: pathlib.Path, only_counting_step) -> None:
    """With nothing outstanding the step is skipped wholesale, as before."""
    scene_zarr = _scene(tmp_path, n_episodes=2, episode_marks=[1])
    root = zarr.open_group(str(scene_zarr), mode='a')
    root.attrs['preprocessing_steps'] = [1]

    run_preprocessing(scene_zarr)

    assert only_counting_step.processed == []


def test_a_store_predating_episode_marks_is_backfilled(tmp_path: pathlib.Path, only_counting_step) -> None:
    """
    Old stores carry only scene-level marks and must not be reprocessed on sight.

    Without the backfill every archived scene would look entirely unprocessed the first time
    it met this code, and re-run the whole pipeline — a full re-SLAM of everything.
    """
    scene_zarr = _scene(tmp_path, n_episodes=3, episode_marks=None)
    root = zarr.open_group(str(scene_zarr), mode='a')
    root.attrs['preprocessing_steps'] = [1]

    run_preprocessing(scene_zarr)

    assert only_counting_step.processed == []
    seeded = zarr.open_group(str(scene_zarr), mode='r')
    assert seeded['episode_0'].attrs['preprocessing_steps'] == [1]


def test_backfill_does_not_claim_a_newly_appended_episode(tmp_path: pathlib.Path, only_counting_step) -> None:
    """
    The empty list build_pzarr writes is what keeps "new" distinguishable from "unmigrated".

    Once any episode carries a mark the store is keeping per-episode records, so an episode
    holding an explicit ``[]`` is new work and must not be seeded from the scene-level marks.
    """
    scene_zarr = _scene(tmp_path, n_episodes=2, episode_marks=[1])
    root = zarr.open_group(str(scene_zarr), mode='a')
    root.attrs['preprocessing_steps'] = [1]
    new = root.create_group('episode_2')
    new.attrs['session_dir'] = 'session_2'
    new.attrs['preprocessing_steps'] = []

    run_preprocessing(scene_zarr)

    assert only_counting_step.processed == ['episode_2']


def test_force_reprocesses_everything(tmp_path: pathlib.Path, only_counting_step) -> None:
    """--force keeps meaning "redo it all", per-episode marks included."""
    scene_zarr = _scene(tmp_path, n_episodes=3, episode_marks=[1])
    root = zarr.open_group(str(scene_zarr), mode='a')
    root.attrs['preprocessing_steps'] = [1]

    run_preprocessing(scene_zarr, force=True)

    assert only_counting_step.processed == ['episode_0', 'episode_1', 'episode_2']


def test_a_scene_level_step_settles_every_episode(tmp_path: pathlib.Path, only_counting_step) -> None:
    """
    A step that finishes in prepare_scene marks the episodes it declined to visit.

    so-align works this way — one transform fit for the whole scene, nothing per-episode. If
    StepComplete left the episodes unmarked they would read as outstanding forever and the
    step would be re-entered on every single run.
    """
    scene_zarr = _scene(tmp_path, n_episodes=2, episode_marks=[])
    only_counting_step.complete_early = True

    run_preprocessing(scene_zarr)

    root = zarr.open_group(str(scene_zarr), mode='r')
    assert root.attrs['preprocessing_steps'] == [1]
    assert root['episode_0'].attrs['preprocessing_steps'] == [1]
    assert root['episode_1'].attrs['preprocessing_steps'] == [1]


def test_a_failed_episode_is_not_marked_done(tmp_path: pathlib.Path, only_counting_step) -> None:
    """A failure keeps the episode retryable under --force rather than looking finished."""

    class _FailingStep(_CountingStep):
        def process_episode(self, scene: SceneContext, episode: Episode) -> None:
            raise RuntimeError('nope')

    scene_zarr = _scene(tmp_path, n_episodes=1, episode_marks=[])
    _FailingStep().run_step(scene_zarr)

    ep = zarr.open_group(str(scene_zarr), mode='r')['episode_0']
    assert ep.attrs['preprocessing_steps'] == []
    assert ep.attrs['failure']['step'] == 'counting-step'
