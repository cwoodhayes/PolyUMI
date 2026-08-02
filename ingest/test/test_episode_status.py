"""Tests for per-episode failure flagging (episode_status) and the step harness that uses it."""

from __future__ import annotations

import json
import pathlib

import pytest
import zarr
from polyumi_ingest.episode_status import Episode, SceneContext, episode_guard
from polyumi_ingest.manifests import SceneManifest, set_episode_unusable
from polyumi_ingest.preproc.step_base import PreprocessingStep, StepComplete


def _make_scene(tmp_path: pathlib.Path, n_episodes: int = 3) -> pathlib.Path:
    """Build a minimal scene.zarr with ``n_episodes`` episode groups and return its path."""
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    for i in range(n_episodes):
        ep = root.require_group(f'episode_{i}')
        ep.attrs['session_type'] = 'MAPPING' if i == 0 else 'EPISODE'
        ep.attrs['session_dir'] = f'session_{i}'
    return scene_zarr


def _episode(scene_zarr: pathlib.Path, index: int) -> Episode:
    root = zarr.open_group(str(scene_zarr), mode='a')
    return Episode.from_key(root, f'episode_{index}')


def _unusable(scene_dir: pathlib.Path) -> list[str]:
    manifest = SceneManifest.from_scene_dir(scene_dir)
    return manifest.unusable_episodes if manifest else []


def test_guard_records_failure_in_zarr_and_scene_json(tmp_path: pathlib.Path) -> None:
    """A raising episode is flagged in its group and named in scene.json, without propagating."""
    scene_zarr = _make_scene(tmp_path)
    episode = _episode(scene_zarr, 1)

    with episode_guard(episode, tmp_path, step='demo-step'):
        raise ValueError('frame 75 is empty')

    failure = _episode(scene_zarr, 1).failure
    assert failure is not None
    assert failure.step == 'demo-step'
    assert 'ValueError: frame 75 is empty' == failure.error
    assert failure.failed_at  # ISO timestamp, whatever the clock said
    assert _unusable(tmp_path) == ['session_1']


def test_guard_leaves_a_healthy_episode_untouched(tmp_path: pathlib.Path) -> None:
    """The happy path writes no failure record and no scene.json entry."""
    scene_zarr = _make_scene(tmp_path)

    with episode_guard(_episode(scene_zarr, 1), tmp_path, step='demo-step'):
        pass

    assert _episode(scene_zarr, 1).failure is None
    assert _unusable(tmp_path) == []


def test_guard_clears_both_marks_on_a_successful_retry(tmp_path: pathlib.Path) -> None:
    """A retry that succeeds un-flags the episode and removes the entry this module added."""
    scene_zarr = _make_scene(tmp_path)

    with episode_guard(_episode(scene_zarr, 1), tmp_path, step='demo-step'):
        raise RuntimeError('transient')
    assert _unusable(tmp_path) == ['session_1']

    with episode_guard(_episode(scene_zarr, 1), tmp_path, step='demo-step'):
        pass

    assert _episode(scene_zarr, 1).failure is None
    assert _unusable(tmp_path) == []


def test_guard_does_not_unmark_a_human_marked_episode(tmp_path: pathlib.Path) -> None:
    """
    An episode a human marked unusable stays marked when a step later succeeds on it.

    The zarr-side ``failure`` record is what proves an entry was machine-added; a hand-marked
    episode has none, so nothing here is entitled to clear it.
    """
    scene_zarr = _make_scene(tmp_path)
    SceneManifest(scene_id='scene-x', unusable_episodes=['session_1']).write_to_scene_dir(tmp_path)

    with episode_guard(_episode(scene_zarr, 1), tmp_path, step='demo-step'):
        pass

    assert _unusable(tmp_path) == ['session_1']


def test_guard_lets_keyboard_interrupt_through(tmp_path: pathlib.Path) -> None:
    """Ctrl-C means the operator wants out, not that this episode is bad."""
    scene_zarr = _make_scene(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        with episode_guard(_episode(scene_zarr, 1), tmp_path, step='demo-step'):
            raise KeyboardInterrupt

    assert _episode(scene_zarr, 1).failure is None
    assert _unusable(tmp_path) == []


def test_scene_context_orders_episodes_numerically(tmp_path: pathlib.Path) -> None:
    """episode_10 sorts after episode_9, which plain string sorting gets wrong."""
    scene_zarr = _make_scene(tmp_path, n_episodes=12)
    scene = SceneContext.open(scene_zarr)
    assert [ep.index for ep in scene.episodes] == list(range(12))
    assert scene.episodes[0].is_mapping
    assert not scene.episodes[1].is_mapping


def test_set_episode_unusable_creates_manifest_from_session_metadata(tmp_path: pathlib.Path) -> None:
    """With no scene.json yet, the created one takes its identity from a session's metadata."""
    session_dir = tmp_path / 'session_0'
    session_dir.mkdir()
    (session_dir / 'metadata.json').write_text(json.dumps({'scene_id': 'abc-123', 'task': 'pick mug'}))

    assert set_episode_unusable(tmp_path, 'session_0', True) is True

    manifest = SceneManifest.from_scene_dir(tmp_path)
    assert manifest is not None
    assert manifest.scene_id == 'abc-123'
    assert manifest.task == 'pick mug'
    assert manifest.unusable_episodes == ['session_0']

    # Idempotent: marking an already-marked episode reports no change.
    assert set_episode_unusable(tmp_path, 'session_0', True) is False
    assert set_episode_unusable(tmp_path, 'session_0', False) is True
    assert _unusable(tmp_path) == []


def test_set_episode_unusable_preserves_other_manifest_fields(tmp_path: pathlib.Path) -> None:
    """Flagging an episode must not disturb notes, task, or pose-source overrides."""
    SceneManifest(
        scene_id='scene-y',
        task='fold towel',
        notes='second attempt',
        pose_source_overrides={'session_2': 'optitrack'},
    ).write_to_scene_dir(tmp_path)

    set_episode_unusable(tmp_path, 'session_1', True)

    manifest = SceneManifest.from_scene_dir(tmp_path)
    assert manifest is not None
    assert (manifest.task, manifest.notes) == ('fold towel', 'second attempt')
    assert manifest.pose_source_overrides == {'session_2': 'optitrack'}
    assert manifest.unusable_episodes == ['session_1']


# ---------------------------------------------------------------------------
# The step harness
# ---------------------------------------------------------------------------


class _DemoStep(PreprocessingStep):
    """Records which hooks ran; fails on whichever episode keys it's told to."""

    step_number = -1
    step_name = 'demo-step'

    def __init__(self, fail_on: set[str] | None = None, complete_early: bool = False) -> None:
        self.fail_on = fail_on or set()
        self.complete_early = complete_early
        self.prepared = 0
        self.processed: list[str] = []
        self.finished = 0

    def prepare_scene(self, scene: SceneContext) -> None:
        self.prepared += 1
        if self.complete_early:
            raise StepComplete('nothing to do here.')

    def process_episode(self, scene: SceneContext, episode: Episode) -> None:
        self.processed.append(episode.key)
        if episode.key in self.fail_on:
            raise RuntimeError(f'{episode.key} is broken')

    def finish_scene(self, scene: SceneContext) -> None:
        self.finished += 1


def test_harness_isolates_a_failing_episode(tmp_path: pathlib.Path) -> None:
    """One bad episode is flagged; its siblings and the reduce phase still run."""
    scene_zarr = _make_scene(tmp_path)
    step = _DemoStep(fail_on={'episode_1'})

    step.run_step(scene_zarr)

    assert step.processed == ['episode_0', 'episode_1', 'episode_2']
    assert step.finished == 1
    assert _episode(scene_zarr, 1).failure is not None
    assert _episode(scene_zarr, 0).failure is None
    assert _episode(scene_zarr, 2).failure is None
    assert _unusable(tmp_path) == ['session_1']


def test_harness_skips_an_already_flagged_episode(tmp_path: pathlib.Path) -> None:
    """A later step doesn't retry what an earlier one already gave up on."""
    scene_zarr = _make_scene(tmp_path)
    _DemoStep(fail_on={'episode_1'}).run_step(scene_zarr)

    later = _DemoStep()
    later.run_step(scene_zarr)

    assert later.processed == ['episode_0', 'episode_2']
    assert _unusable(tmp_path) == ['session_1']


def test_harness_force_retries_a_flagged_episode(tmp_path: pathlib.Path) -> None:
    """--force re-attempts flagged episodes, clearing both marks when they now succeed."""
    scene_zarr = _make_scene(tmp_path)
    _DemoStep(fail_on={'episode_1'}).run_step(scene_zarr)

    retry = _DemoStep()
    retry.run_step(scene_zarr, force=True)

    assert retry.processed == ['episode_0', 'episode_1', 'episode_2']
    assert _episode(scene_zarr, 1).failure is None
    assert _unusable(tmp_path) == []


def test_harness_force_retry_that_fails_again_stays_flagged(tmp_path: pathlib.Path) -> None:
    """A still-broken episode simply re-flags itself, under the step that just failed."""
    scene_zarr = _make_scene(tmp_path)
    _DemoStep(fail_on={'episode_1'}).run_step(scene_zarr)

    _DemoStep(fail_on={'episode_1'}).run_step(scene_zarr, force=True)

    failure = _episode(scene_zarr, 1).failure
    assert failure is not None
    assert failure.step == 'demo-step'
    assert _unusable(tmp_path) == ['session_1']


def test_harness_step_complete_skips_the_remaining_phases(tmp_path: pathlib.Path) -> None:
    """StepComplete from prepare_scene ends the step without touching episodes."""
    scene_zarr = _make_scene(tmp_path)
    step = _DemoStep(complete_early=True)

    step.run_step(scene_zarr)

    assert (step.prepared, step.processed, step.finished) == (1, [], 0)


def test_harness_scene_level_failure_still_raises(tmp_path: pathlib.Path) -> None:
    """A prepare_scene failure isn't episode-shaped, so it fails the whole step."""

    class _BadPrepare(_DemoStep):
        def prepare_scene(self, scene: SceneContext) -> None:
            raise RuntimeError('calibration file is missing')

    with pytest.raises(RuntimeError, match='calibration file is missing'):
        _BadPrepare().run_step(_make_scene(tmp_path))


def test_harness_rejects_a_missing_store(tmp_path: pathlib.Path) -> None:
    """Opening for mutation would create an empty store; refuse instead."""
    with pytest.raises(FileNotFoundError):
        _DemoStep().run_step(tmp_path / 'nope.zarr')
    assert not (tmp_path / 'nope.zarr').exists()
