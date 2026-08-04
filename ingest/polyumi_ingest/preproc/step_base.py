"""Shared preprocessing step base class and pipeline helpers."""

from __future__ import annotations

import datetime as dt
import logging
import pathlib
import shutil
from abc import ABC
from typing import TypeVar

import zarr

_PS = TypeVar('_PS', bound='PreprocessingStep')

from polyumi_ingest.episode_status import Episode, SceneContext, episode_guard
from polyumi_ingest.gitinfo import git_sha
from polyumi_ingest.pzarr.scene_files import SceneFiles
from polyumi_ingest.pzarr.version import PZARR_VERSION

log = logging.getLogger(__name__)

PREPROCESSING_STEPS: dict[int, type[PreprocessingStep]] = {}


def register_preprocessing_step(step_number: int, step_name: str):
    """Register a preprocessing step class with explicit metadata."""

    def decorator(cls: type[_PS]) -> type[_PS]:
        if step_number in PREPROCESSING_STEPS:
            raise ValueError(f'Duplicate preprocessing step: {step_number}')
        cls.step_number = step_number  # type: ignore[attr-defined]
        cls.step_name = step_name  # type: ignore[attr-defined]
        PREPROCESSING_STEPS[step_number] = cls
        return cls

    return decorator


def available_preprocessing_steps() -> list[type[PreprocessingStep]]:
    """Return registered preprocessing steps in execution order."""
    return [PREPROCESSING_STEPS[k] for k in sorted(PREPROCESSING_STEPS)]


def _scene_dirs(recordings_dir: pathlib.Path) -> list[pathlib.Path]:
    recordings_dir = recordings_dir.resolve()
    if not recordings_dir.is_dir():
        raise FileNotFoundError(f'Recordings directory not found: {recordings_dir}')
    return sorted(p for p in recordings_dir.iterdir() if p.is_dir() and p.name.startswith('scene_'))


def preprocessing_steps_done(root: zarr.Group) -> list[int]:
    """Return the sorted list of preprocessing step numbers already recorded on ``root``."""
    raw = root.attrs.get('preprocessing_steps', [])
    if not isinstance(raw, list):
        return []
    try:
        return [int(step) for step in raw if isinstance(step, (int, float, str))]
    except (ValueError, TypeError):
        log.warning(f'Invalid preprocessing_steps attribute: {raw}')
        return []


def preprocessing_step_versions(root: zarr.Group) -> dict[str, dict]:
    """
    Return ``{step_number_as_str: {'git_sha': ..., 'completed_at': ...}}`` recorded on ``root``.

    Empty for any store last processed before this provenance was recorded, so callers must
    treat a missing entry as "unknown", not as "not run" — ``preprocessing_steps`` remains
    the authority on which steps are complete.
    """
    raw = root.attrs.get('preprocessing_step_versions', {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _mark_preprocessing_step(root: zarr.Group, step_number: int) -> None:
    steps = preprocessing_steps_done(root)
    if step_number not in steps:
        steps.append(step_number)
        steps.sort()
    root.attrs['preprocessing_steps'] = steps
    # Per-step provenance, alongside the root's build-time `git_sha`: steps are re-run
    # individually and often under a later commit than the one that built the store, so a
    # single store-level sha can't say which code produced any particular step's output.
    # Keys are strings because zarr attrs round-trip through JSON, which has no int keys.
    versions = preprocessing_step_versions(root)
    versions[str(step_number)] = {
        'git_sha': git_sha(),
        'completed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    root.attrs['preprocessing_step_versions'] = versions


def _write_scalar(group: zarr.Group, name: str, value: float | int) -> None:
    """Write a scalar annotation as a group attribute."""
    group.attrs[name] = value


def _warn_if_outdated_pzarr(root: zarr.Group, scene_label: str) -> None:
    """
    Log if a store's schema version doesn't match the running code. Never blocks.

    An older store still processes fine — the steps overwrite what they own — but its
    untouched outputs may predate the current schema, which matters most when several scenes
    are combined into one dataset. A *newer* store is the more dangerous direction: the code
    reading it can't know what it doesn't know, so that warns harder.
    """
    stored = int(root.attrs.get('pzarr_version', 1))  # pre-v1 stores have no attr
    if stored == PZARR_VERSION:
        return
    if stored < PZARR_VERSION:
        log.warning(
            f'{scene_label}: built under pzarr v{stored}, current is v{PZARR_VERSION} — outputs may '
            f'predate the current schema; a full `pingest pp --force` reprocesses and restamps it.'
        )
    else:
        log.error(
            f'{scene_label}: written by pzarr v{stored}, but this code only knows v{PZARR_VERSION} — '
            f'it may read fields that have since changed meaning. Update your checkout.'
        )


class StepComplete(Exception):  # noqa: N818 — control flow, not an error
    """
    Raised by ``prepare_scene`` to end a step early with nothing left to do.

    Not a failure: the harness logs the message and returns, and the step is still marked
    complete. Used for "output already present, use --force" and for the degenerate inputs a
    step answers at scene level (so-align storing an identity transform when there's no
    OptiTrack data to align to).
    """


class PreprocessingStep(ABC):
    """
    Base class for a single preprocessing step, and the harness that runs one.

    Steps are map-reduce over a scene's episodes: :meth:`prepare_scene` once, then
    :meth:`process_episode` for each episode, then :meth:`finish_scene` once. All three
    default to no-ops, so a step implements only the phases it needs. ``run_step`` is the
    harness itself and is not meant to be overridden.

    The point of the shape is fault isolation. An exception from ``process_episode`` flags
    *that* episode unusable (in the store and in ``scene.json``) and the scene carries on —
    a single corrupt session can't discard the work already done on its siblings. Scene-level
    hooks have no such net: if ``prepare_scene`` or ``finish_scene`` raises, the step fails,
    which is right, because their failure isn't episode-shaped.

    A step instance handles exactly one scene (``run_preprocessing`` constructs one per scene),
    so ``prepare_scene`` may stash whatever the later hooks need on ``self``.
    """

    step_number: int
    step_name: str

    def prepare_scene(self, scene: SceneContext) -> None:
        """
        Scene-level work before any episode: load config, build shared artifacts.

        Raise :class:`StepComplete` to finish the step here, skipping both later phases.
        """

    def process_episode(self, scene: SceneContext, episode: Episode) -> None:
        """
        Work on one episode. Raising flags it unusable; the rest of the scene continues.

        Returning early is *not* a failure and flags nothing — that's how a step declines an
        episode it has nothing to do for (a missing prerequisite, the mapping session).
        """

    def finish_scene(self, scene: SceneContext) -> None:
        """Scene-level work after every episode: the reduce half of a map-reduce step."""

    def run_step(self, scene_zarr: pathlib.Path, force: bool = False) -> None:
        """Run this step's three phases over a scene.zarr, isolating per-episode failures."""
        # Checked here as well as in run(): opening for mutation would otherwise *create* an
        # empty store at this path and then complain that it has no episodes.
        if not scene_zarr.exists():
            raise FileNotFoundError(f'No scene.zarr found at {scene_zarr}')
        scene = SceneContext.open(scene_zarr, force=force)
        episodes = scene.episodes
        if not episodes:
            raise RuntimeError(f'No episodes found in {scene_zarr}')

        try:
            self.prepare_scene(scene)
        except StepComplete as done:
            log.info(f'{self.step_name}: {done}')
            return

        for episode in episodes:
            failure = episode.failure
            if failure is not None and not force:
                log.warning(
                    f'{episode.key}: skipping — flagged unusable in {failure.step} ({failure.error}); --force to retry'
                )
                continue
            if failure is not None:
                log.info(f'{episode.key}: retrying (was flagged in {failure.step})')
            with episode_guard(episode, scene.scene_dir, step=self.step_name):
                self.process_episode(scene, episode)

        self.finish_scene(scene)

    def run(self, scene_path: pathlib.Path, copy: bool = False, force: bool = False) -> pathlib.Path:
        """Run the step on a scene directory or scene.zarr path."""
        scene_zarr = SceneFiles.resolve_zarr_path(scene_path)
        if not scene_zarr.exists():
            raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

        target_zarr = scene_zarr
        if copy:
            target_zarr = scene_zarr.parent / f'scene_pp{self.step_number}.zarr'
            if target_zarr.exists():
                if not force:
                    raise FileExistsError(f'Preprocessed scene already exists: {target_zarr}')
                shutil.rmtree(target_zarr)
            shutil.copytree(scene_zarr, target_zarr)

        self.run_step(target_zarr, force=force)
        root = zarr.open_group(str(target_zarr), mode='a')
        _mark_preprocessing_step(root, self.step_number)
        return target_zarr


def run_preprocessing(
    scene_path: pathlib.Path,
    step_number: int | None = None,
    copy: bool = False,
    force: bool = False,
) -> pathlib.Path:
    """Run one preprocessing step or the full pipeline on a scene."""
    scene_zarr = SceneFiles.resolve_zarr_path(scene_path)
    if not scene_zarr.exists():
        raise FileNotFoundError(f'No scene.zarr found at {scene_path}')

    root = zarr.open_group(str(scene_zarr), mode='a')
    _warn_if_outdated_pzarr(root, scene_zarr.parent.name)
    completed_steps = set(preprocessing_steps_done(root))
    step_numbers = [step_number] if step_number is not None else sorted(PREPROCESSING_STEPS)

    current_path = scene_path
    ran: set[int] = set()
    for number in step_numbers:
        try:
            step_cls = PREPROCESSING_STEPS[number]
        except KeyError:
            raise KeyError(f'Unknown preprocessing step: {number}')
        if number in completed_steps and not force:
            log.info(f'Skipping {scene_zarr.name}: step {number} already complete')
            continue
        log.info(f'Running step {number} ({step_cls.step_name}) on {scene_zarr.name}')
        step = step_cls()
        current_path = step.run(current_path, copy=copy, force=force)
        scene_zarr = SceneFiles.resolve_zarr_path(current_path)
        root = zarr.open_group(str(scene_zarr), mode='a')
        _mark_preprocessing_step(root, number)
        ran.add(number)
        completed_steps = set(preprocessing_steps_done(root))
        if step_number is None:
            copy = False

    # Restamp only when every registered step actually ran *here*. Trusting
    # `preprocessing_steps_done` instead would stamp the current version onto a store whose
    # skipped steps still hold output from an older schema — worse than not stamping at all,
    # since the stamp is what tells the next run whether it can believe what it reads.
    if ran == set(PREPROCESSING_STEPS) and int(root.attrs.get('pzarr_version', 1)) != PZARR_VERSION:
        log.info(f'{scene_zarr.parent.name}: whole pipeline re-run; restamping as pzarr v{PZARR_VERSION}')
        root.attrs['pzarr_version'] = PZARR_VERSION

    return SceneFiles.resolve_zarr_path(current_path)


def run_preprocessing_on_recordings(
    recordings_dir: pathlib.Path,
    step_number: int | None = None,
    copy: bool = False,
    force: bool = False,
) -> list[pathlib.Path]:
    """
    Run preprocessing on every scene under recordings_dir.

    A scene that fails outright (as opposed to one episode failing, which the step harness
    already absorbs) is logged and skipped so the rest of the batch still runs; the failures
    are summarised at the end. Same reasoning as the per-episode guard, one level up.
    """
    outputs: list[pathlib.Path] = []
    failures: list[tuple[str, str]] = []
    for scene_dir in _scene_dirs(recordings_dir):
        zarr_path = SceneFiles.resolve_zarr_path(scene_dir)
        if not zarr_path.exists():
            log.info(f'Skipping {scene_dir.name}: no scene.zarr found')
            continue
        try:
            outputs.append(run_preprocessing(scene_dir, step_number=step_number, copy=copy, force=force))
        except Exception as exc:
            log.debug(f'{scene_dir.name}: traceback', exc_info=True)
            log.error(f'{scene_dir.name} failed, continuing with the remaining scenes: {exc}')
            failures.append((scene_dir.name, f'{type(exc).__name__}: {exc}'))

    if failures:
        log.error(f'{len(failures)} scene(s) failed:')
        for name, reason in failures:
            log.error(f'  {name}: {reason}')
    return outputs
