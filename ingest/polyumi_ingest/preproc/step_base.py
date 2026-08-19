"""Shared preprocessing step base class and pipeline helpers."""

from __future__ import annotations

import datetime as dt
import logging
import pathlib
import shutil
from abc import ABC
from collections.abc import Iterable
from typing import TypeVar

import zarr

_PS = TypeVar('_PS', bound='PreprocessingStep')

from polyumi_ingest.episode_status import FAILURE_ATTR, Episode, SceneContext, episode_guard, episode_keys
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


#: Steps whose marks were cleared by :func:`_invalidate_downstream_steps` and which must
#: therefore *recompute*, not merely re-run.
#:
#: Clearing the marks is not enough on its own. Several steps carry their own
#: "output already present; use --force to recompute" guard (so_align's ``StepComplete``,
#: eef-pose's per-episode check), so an invalidated step would be re-entered by the harness
#: and then decline to do anything — which is exactly what happened on the first attempt to
#: rebuild scene 31d6: all 63 episodes logged "eef/pose_* already present" and the stale
#: poses survived a run that looked, from the outside, like a successful rebuild.
#:
#: Persisted on the store rather than held in memory because the invalidation and the rebuild
#: are routinely different processes: `pingest pp 2 --force` today, `pingest pp` tomorrow.
STALE_STEPS_ATTR = 'preprocessing_steps_needing_rebuild'


def steps_needing_rebuild(root: zarr.Group) -> set[int]:
    """Return step numbers invalidated by an earlier run and not yet rebuilt."""
    raw = root.attrs.get(STALE_STEPS_ATTR, [])
    if not isinstance(raw, list):
        return set()
    out = set()
    for step in raw:
        try:
            out.add(int(step))
        except (TypeError, ValueError):
            continue
    return out


def _set_steps_needing_rebuild(root: zarr.Group, steps: set[int]) -> None:
    """Persist the set of steps still awaiting a rebuild."""
    root.attrs[STALE_STEPS_ATTR] = sorted(steps)


def _invalidate_downstream_steps(root: zarr.Group, step_number: int) -> list[int]:
    """
    Drop the completion marks for every step after ``step_number``. Returns what was cleared.

    Steps form a chain — each reads what the ones before it wrote — so re-running step N makes
    the output of every later step describe inputs that no longer exist. Nothing in the store
    said so: the marks are a flat set of "done" numbers with no dependency order, so a re-run
    of one step left its successors looking complete forever.

    That is not hypothetical. Scene 31d6 had step 2 re-run under the gripper-mask fix 90
    minutes *after* step 5 last ran, so ``eef/pose_slam`` — the array DP export reads — was
    still the retarget of the previous SLAM pass. It exported 18 episodes (17% of the dataset)
    of poses belonging to a superseded trajectory, and every frame count and tracking ratio
    alongside them described the new one.

    Only marks are cleared, never data: the later steps' arrays stay until those steps re-run
    and overwrite them. Per-episode marks go too, otherwise ``run_step``'s own
    already-done check would skip every episode of the steps this just re-opened.

    Note this cascades honestly rather than cheaply — re-running step 1 invalidates step 2,
    which means a full re-SLAM. That is the true cost of changing the time sync every later
    step is aligned to, and it is rare; the common case (re-running SLAM) only invalidates the
    cheap steps after it.
    """
    done = preprocessing_steps_done(root)
    later = sorted(step for step in done if step > step_number)
    if not later:
        return []

    root.attrs['preprocessing_steps'] = [step for step in done if step <= step_number]
    versions = preprocessing_step_versions(root)
    for step in later:
        versions.pop(str(step), None)
    root.attrs['preprocessing_step_versions'] = versions

    for key in episode_keys(root):
        ep_steps = episode_steps_done(root[key])
        if ep_steps is None:
            continue
        kept = [step for step in ep_steps if step <= step_number]
        if len(kept) != len(ep_steps):
            root[key].attrs['preprocessing_steps'] = kept

    _set_steps_needing_rebuild(root, steps_needing_rebuild(root) | set(later))
    return later


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

    invalidated = _invalidate_downstream_steps(root, step_number)
    if invalidated:
        log.warning(
            f'step {step_number} re-ran, so step(s) {invalidated} now describe inputs that no '
            f'longer exist and have been marked incomplete — run `pingest pp` to rebuild them.'
        )


def episode_steps_done(ep_grp: zarr.Group) -> list[int] | None:
    """
    Return the steps recorded on one episode group, or None if it has never been marked.

    None and ``[]`` mean different things and callers depend on it: None is a store written
    before per-episode marks existed, whose completion is only known at scene level, while
    ``[]`` is an episode that genuinely has no step done yet.
    """
    raw = ep_grp.attrs.get('preprocessing_steps')
    if not isinstance(raw, list):
        return None
    try:
        return [int(step) for step in raw if isinstance(step, (int, float, str))]
    except (ValueError, TypeError):
        log.warning(f'Invalid per-episode preprocessing_steps attribute: {raw}')
        return []


def _mark_episode_step(ep_grp: zarr.Group, step_number: int) -> None:
    """Record that a step has been completed for one episode."""
    steps = episode_steps_done(ep_grp) or []
    if step_number not in steps:
        steps.append(step_number)
        steps.sort()
    ep_grp.attrs['preprocessing_steps'] = steps


def _backfill_episode_steps(root: zarr.Group) -> None:
    """
    Seed per-episode marks from the scene-level ones on stores that predate them.

    Without this every pre-existing scene would look like it had no episode finished, and the
    gate below would re-run the whole pipeline — a full re-SLAM of everything already
    processed.

    Decided per episode, not for the store as a whole. build_pzarr writes an explicit empty
    list on every episode it creates — including appended ones — so a *missing* attr always
    means an episode from before per-episode marks, whatever its neighbours look like. Judging
    the store as a whole got this backwards on the case that matters most: appending one
    session to a pre-marks store gave that store a marked episode, which suppressed the
    backfill for every old episode and re-ran the whole pipeline on all of them.
    """
    completed = preprocessing_steps_done(root)
    if not completed:
        return
    for key in episode_keys(root):
        if episode_steps_done(root[key]) is None:
            root[key].attrs['preprocessing_steps'] = sorted(completed)


def clear_step_marks(root: zarr.Group, step_numbers: Iterable[int] | None = None) -> None:
    """
    Drop the completion marks for the given steps (all recorded ones if None).

    Both levels, root and per-episode, or the clear is worse than useless: per-episode marks
    gate the episode loop, so clearing only the root's leaves a forced run that died part-way
    with its later steps' *stale* episode marks intact. The next non-forced run then sees the
    root mark gone (so it runs the step), finds nothing outstanding (so it skips every
    episode), and stamps the step complete having computed nothing.

    Called up front for the whole pipeline rather than per step as each begins, so a run that
    dies at step 2 stops advertising steps 3-5 — whose outputs were computed from the
    *previous* step 2 — and so the catalog's progress display drops to 0/N for the duration
    instead of sitting at N/N until the last step finishes.
    """
    recorded = set(preprocessing_steps_done(root))
    drop = recorded if step_numbers is None else set(step_numbers)
    drop_keys = {str(n) for n in drop}
    root.attrs['preprocessing_steps'] = sorted(recorded - drop)
    root.attrs['preprocessing_step_versions'] = {
        k: v for k, v in preprocessing_step_versions(root).items() if k not in drop_keys
    }
    for key in episode_keys(root):
        # None means an episode from before per-episode marks; _backfill_episode_steps has
        # already run by the time a forced run gets here, so one still unmarked has nothing
        # to clear — writing [] would claim it was newly built, which it wasn't.
        steps = episode_steps_done(root[key])
        if steps is not None:
            root[key].attrs['preprocessing_steps'] = sorted(set(steps) - drop)


def _episodes_missing_step(root: zarr.Group, step_number: int) -> list[str]:
    """
    Return episodes that still need ``step_number``, ignoring ones flagged unusable.

    Failed episodes are excluded because the harness skips them anyway without ``--force``;
    counting them would leave every step permanently "incomplete" for the whole scene.
    """
    missing = []
    for key in episode_keys(root):
        ep_grp = root[key]
        if isinstance(ep_grp.attrs.get(FAILURE_ATTR), dict):
            continue
        if step_number not in (episode_steps_done(ep_grp) or []):
            missing.append(key)
    return missing


def _write_scalar(group: zarr.Group, name: str, value: float | int) -> None:
    """Write a scalar annotation as a group attribute."""
    group.attrs[name] = value


def _stored_pzarr_version(root: zarr.Group) -> int | None:
    """
    Read a store's ``pzarr_version``, or None if it isn't a number.

    Missing means a pre-v1 store, which is v1 by definition. Present-but-unparseable means a
    corrupt or hand-edited attr; the callers below only warn and restamp, so neither should
    abort a preprocessing run over it — hence None rather than a raise.
    """
    raw = root.attrs.get('pzarr_version', 1)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _warn_if_outdated_pzarr(root: zarr.Group, scene_label: str) -> None:
    """
    Log if a store's schema version doesn't match the running code. Never blocks.

    An older store still processes fine — the steps overwrite what they own — but its
    untouched outputs may predate the current schema, which matters most when several scenes
    are combined into one dataset. A *newer* store is the more dangerous direction: the code
    reading it can't know what it doesn't know, so that warns harder.
    """
    stored = _stored_pzarr_version(root)
    if stored is None:
        log.warning(
            f'{scene_label}: pzarr_version attr is {root.attrs.get("pzarr_version")!r}, not a version '
            f'number — cannot tell whether this store matches pzarr v{PZARR_VERSION}.'
        )
        return
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


def _completion_times(root: zarr.Group) -> dict[int, dt.datetime]:
    """Return the parsed ``completed_at`` per completed step, omitting unusable stamps."""
    versions = preprocessing_step_versions(root)
    times: dict[int, dt.datetime] = {}
    for step in preprocessing_steps_done(root):
        raw = versions.get(str(step), {}).get('completed_at')
        if not isinstance(raw, str):
            continue
        try:
            stamp = dt.datetime.fromisoformat(raw)
        except ValueError:
            continue
        # Naive stamps can't be compared against aware ones; everything is written in UTC.
        times[step] = stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)
    return times


def _stale_steps(root: zarr.Group) -> list[int]:
    """
    Completed steps that finished *before* a step they depend on. Empty on a healthy store.

    Retroactive detection for stores written before :func:`_invalidate_downstream_steps`
    existed, which is the only reason a scene can be in this state going forward. Steps
    without a ``completed_at`` are invisible here — a missing stamp is "unknown", not
    "stale" — so pre-provenance stores are left alone rather than needlessly reprocessed.
    """
    times = _completion_times(root)
    stale: list[int] = []
    latest_upstream: dt.datetime | None = None
    for step in sorted(times):
        if latest_upstream is not None and times[step] < latest_upstream:
            stale.append(step)
        elif latest_upstream is None or times[step] > latest_upstream:
            latest_upstream = times[step]
    return stale


class StepComplete(Exception):  # noqa: N818 — control flow, not an error
    """
    Raised by ``prepare_scene`` to end a step early with nothing left to do.

    Not a failure: the harness logs the message and returns, and the step is still marked
    complete. Used for "output already present, use --force" and for the degenerate inputs a
    step answers at scene level (so-align storing an identity transform when there's no
    OptiTrack data to align to).

    Only for steps whose answer is genuinely scene-wide, because the harness marks *every*
    episode done on the way out — including ones appended since the last run, which it never
    looked at. so-align qualifies: its output is one T_ws for the whole scene, so a new
    episode needs nothing computed for it. A step that raised this because its own per-episode
    output "already exists" would silently skip the episodes that have none.
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
            # The step settled the whole scene at once (so-align's single T_ws, an output
            # already present). Mark every episode so it doesn't read as outstanding work and
            # get re-attempted on every later run.
            for episode in episodes:
                _mark_episode_step(episode.group, self.step_number)
            return

        for episode in episodes:
            if not force and self.step_number in (episode_steps_done(episode.group) or []):
                log.info(f'{episode.key}: skipping — step {self.step_number} already done for this episode')
                continue
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
            # Only a clean pass counts. A failed episode keeps its `failure` attr instead, so
            # --force retries it rather than the next run treating it as finished.
            if episode.failure is None:
                _mark_episode_step(episode.group, self.step_number)

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
    _backfill_episode_steps(root)
    # Heal a store whose steps were run out of order before the invalidation above existed.
    # Clearing from the *earliest* stale step keeps the chain intact — reprocessing step 5
    # while step 3 is still stale would just rebuild it from stale inputs.
    #
    # Before the force-clear below, not after: this reads `completed_at` stamps, and clearing
    # the marks takes those stamps with them. A forced run would otherwise heal silently and
    # never say which steps were out of order.
    stale = _stale_steps(root)
    if stale:
        log.warning(
            f'{scene_zarr.parent.name}: step(s) {stale} completed before a step they depend on '
            f'— their output describes superseded inputs. Marking them incomplete so this run '
            f'rebuilds them.'
        )
        _invalidate_downstream_steps(root, min(stale) - 1)
    step_numbers = [step_number] if step_number is not None else sorted(PREPROCESSING_STEPS)
    if force:
        # Exactly the steps about to re-run, and at both levels. Without this a forced run
        # that dies part-way leaves stale per-episode marks that make the next non-forced run
        # skip the work and mark it done anyway. The catalog clears the same marks up front
        # for its progress display; this is what makes the CLI's --force honest too.
        clear_step_marks(root, step_numbers)
    completed_steps = set(preprocessing_steps_done(root))

    current_path = scene_path
    ran: set[int] = set()
    for number in step_numbers:
        try:
            step_cls = PREPROCESSING_STEPS[number]
        except KeyError:
            raise KeyError(f'Unknown preprocessing step: {number}')
        # The scene-level mark alone is not enough: a scene that gained episodes after it was
        # processed is "complete" at scene level while the new episodes have had nothing run
        # on them. The step still runs, but its episode loop skips everything already done.
        outstanding = _episodes_missing_step(root, number)
        # A step whose marks were invalidated must *recompute*, not merely be re-entered:
        # its own "output already present" guard would otherwise decline the work and leave
        # the stale output in place. See STALE_STEPS_ATTR. Checked independently of the marks
        # rather than inferred from them, so the two can't disagree — invalidation normally
        # clears the marks too, but a restored or hand-edited mark must not bury the debt.
        needs_rebuild = number in steps_needing_rebuild(root)
        if number in completed_steps and not force and not outstanding and not needs_rebuild:
            log.info(f'Skipping {scene_zarr.name}: step {number} already complete')
            continue
        if number in completed_steps and outstanding:
            log.info(f'Step {number} complete for {scene_zarr.name} except {len(outstanding)} new episode(s)')
        if needs_rebuild:
            log.info(f'Step {number} was invalidated by an upstream re-run; recomputing it.')
        log.info(f'Running step {number} ({step_cls.step_name}) on {scene_zarr.name}')
        step = step_cls()
        current_path = step.run(current_path, copy=copy, force=force or needs_rebuild)
        scene_zarr = SceneFiles.resolve_zarr_path(current_path)
        root = zarr.open_group(str(scene_zarr), mode='a')
        _mark_preprocessing_step(root, number)
        # Marking invalidates the steps *after* this one, so clear this one's debt afterwards.
        _set_steps_needing_rebuild(root, steps_needing_rebuild(root) - {number})
        ran.add(number)
        completed_steps = set(preprocessing_steps_done(root))
        if step_number is None:
            copy = False

    # Restamp only when every registered step actually ran *here*. Trusting
    # `preprocessing_steps_done` instead would stamp the current version onto a store whose
    # skipped steps still hold output from an older schema — worse than not stamping at all,
    # since the stamp is what tells the next run whether it can believe what it reads.
    # An unparseable stored version (None) restamps too: every step just re-ran, so v4 is
    # what the store now holds regardless of what the corrupt attr claimed.
    if ran == set(PREPROCESSING_STEPS) and _stored_pzarr_version(root) != PZARR_VERSION:
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
