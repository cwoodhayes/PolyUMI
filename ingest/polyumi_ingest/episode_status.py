"""
Per-episode failure records, so one bad episode can't sink a whole scene.

Ingest is a batch pipeline: a scene holds tens of episodes and each preprocessing step walks
all of them. A single damaged session — a zero-byte JPEG left by a killed recorder, a GoPro
file that never finished flushing — used to raise out of the episode loop and abort the
scene, discarding the work already done on every other episode.

Instead, ``build_pzarr`` and the preprocessing-step harness run their per-episode work inside
:func:`episode_guard`, which on failure:

* records it on the episode group (``episode.attrs['failure']``) so later steps skip that
  episode rather than tripping over its half-written arrays, and
* adds the session directory to ``scene.json``'s ``unusable_episodes`` — the marker the
  catalog UI sets by hand and DP export already honours, so the episode drops out of
  datasets with no further work.

Nothing here deletes data: the partial episode group stays on disk for inspection, and a
``--force`` re-run retries it, clearing both marks if it succeeds. Only a mark this module
made is ever cleared — an episode a human marked unusable in the catalog UI has no
``failure`` record, so it stays marked.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import logging
import pathlib
from collections.abc import Iterator

import zarr

from polyumi_ingest.manifests import set_episode_unusable

log = logging.getLogger(__name__)

#: Episode-group attribute holding the :class:`EpisodeFailure` record, if any.
FAILURE_ATTR = 'failure'

MAPPING_SESSION_TYPE = 'MAPPING'


@dataclasses.dataclass(frozen=True)
class EpisodeFailure:
    """Which stage dropped an episode, and why."""

    step: str
    error: str
    failed_at: str


@dataclasses.dataclass(frozen=True)
class Episode:
    """
    One ``episode_N`` group, with the identity every consumer keeps re-deriving.

    ``index`` is parsed from the key rather than being the position in a list: steps skip
    episodes (MAPPING, already-failed), so a position would drift away from the ``episode_N``
    the logs and SLAM temp dirs are named for.
    """

    key: str
    index: int
    group: zarr.Group

    @classmethod
    def from_key(cls, root: zarr.Group, key: str) -> Episode:
        """Build an Episode for ``key`` in a scene root, creating the group if absent."""
        return cls(key=key, index=int(key.split('_')[1]), group=root.require_group(key))

    @property
    def session_type(self) -> str | None:
        """``'EPISODE'`` / ``'MAPPING'`` as recorded by build_pzarr, or None on older stores."""
        value = self.group.attrs.get('session_type')
        return str(value) if isinstance(value, str) else None

    @property
    def session_dir(self) -> str | None:
        """Name of the session directory this episode was built from, if recorded."""
        value = self.group.attrs.get('session_dir')
        return value if isinstance(value, str) and value else None

    @property
    def is_mapping(self) -> bool:
        """True for the scene's mapping session (the SLAM map is built from it)."""
        return self.session_type == MAPPING_SESSION_TYPE

    @property
    def failure(self) -> EpisodeFailure | None:
        """The recorded failure for this episode, or None if it is healthy."""
        raw = self.group.attrs.get(FAILURE_ATTR)
        if not isinstance(raw, dict):
            return None
        return EpisodeFailure(
            step=str(raw.get('step', 'unknown')),
            error=str(raw.get('error', '')),
            failed_at=str(raw.get('failed_at', '')),
        )


def episode_keys(root: zarr.Group) -> list[str]:
    """Return every ``episode_*`` group key in a scene root, in numeric order."""
    keys = [k for k in root.keys() if k.startswith('episode_')]
    return sorted(keys, key=lambda k: int(k.split('_')[1]))


@dataclasses.dataclass
class SceneContext:
    """The scene a preprocessing step is working on, handed to each of its hooks."""

    zarr_path: pathlib.Path
    root: zarr.Group
    force: bool = False

    @classmethod
    def open(cls, zarr_path: pathlib.Path, force: bool = False) -> SceneContext:
        """Open a scene.zarr for mutation and wrap it in a context."""
        return cls(zarr_path=zarr_path, root=zarr.open_group(str(zarr_path), mode='a'), force=force)

    @property
    def scene_dir(self) -> pathlib.Path:
        """The scene directory holding scene.zarr, its sessions, and scene.json."""
        return self.zarr_path.parent

    @property
    def episodes(self) -> list[Episode]:
        """Every episode in the scene, mapping session included, in index order."""
        return [Episode.from_key(self.root, key) for key in episode_keys(self.root)]


def record_episode_failure(
    episode: Episode,
    scene_dir: pathlib.Path,
    step: str,
    exc: BaseException,
) -> EpisodeFailure:
    """Flag an episode as failed in both the zarr store and ``scene.json``."""
    failure = EpisodeFailure(
        step=step,
        error=f'{type(exc).__name__}: {exc}',
        failed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    episode.group.attrs[FAILURE_ATTR] = dataclasses.asdict(failure)

    if episode.session_dir is not None:
        set_episode_unusable(scene_dir, episode.session_dir, True)
        marked = f'; {episode.session_dir} marked unusable in scene.json'
    else:
        # Pre-`session_dir` stores and hand-built fixtures: the zarr flag still keeps the rest
        # of the pipeline off this episode, but there's no key to mark it by outside the store.
        marked = '; no session_dir attr, so not marked in scene.json'
    log.error(f'{episode.key}: {step} failed, continuing with the rest of the scene{marked}: {failure.error}')
    return failure


def clear_episode_failure(episode: Episode, scene_dir: pathlib.Path) -> None:
    """Drop a previously recorded failure after a successful retry, un-marking scene.json."""
    episode.group.attrs.pop(FAILURE_ATTR, None)
    if episode.session_dir is not None:
        set_episode_unusable(scene_dir, episode.session_dir, False)
    log.info(f'{episode.key}: succeeded on retry; no longer flagged unusable.')


@contextlib.contextmanager
def episode_guard(episode: Episode, scene_dir: pathlib.Path, step: str) -> Iterator[None]:
    """
    Run one episode's work; on failure flag it unusable and carry on with the scene.

    Only ``Exception`` is caught. ``KeyboardInterrupt`` and ``SystemExit`` still stop the run,
    since those mean the operator wants out, not that the episode is bad.
    """
    was_failed = episode.failure is not None
    try:
        yield
    except Exception as exc:
        log.debug(f'{episode.key}: {step} traceback', exc_info=True)
        record_episode_failure(episode, scene_dir, step, exc)
    else:
        if was_failed:
            clear_episode_failure(episode, scene_dir)
