"""
Which commits produced a scene's data, for the scene detail pane.

Two independent lineages meet in one scene directory and they are not the same code:

* **Recording** — the ``polyumi_version`` each session's ``metadata.json`` records, which
  ``deploy.sh`` bakes into ``pi/polyumi_pi/_version.py`` at deploy time. This is the Pi's
  commit at capture, frozen the moment the session was written.
* **Preprocessing** — the pzarr root's ``git_sha`` (stamped when the store was built) plus
  the per-step ``preprocessing_step_versions`` written by ``_mark_preprocessing_step``.
  Steps re-run independently and often under a later commit than the build, so the
  per-step shas are what actually answer "which code produced these poses".

Everything here is read from disk on demand rather than cached in the catalog DB, for the
same reason ``pp_status`` and ``episode_quality`` are: these facts live in the pzarr and
in ``metadata.json``, and a step re-run changes them without any sync pass running.
"""

from __future__ import annotations

import json
import logging
import pathlib

log = logging.getLogger('catalog.provenance')

#: How much of a 40-char sha to show in the UI. Long enough to stay unambiguous over this
#: repo's history, short enough not to wrap the detail pane's narrow column.
SHORT_SHA_LEN = 12


def short_sha(sha: str | None) -> str | None:
    """Abbreviate a full commit sha for display, passing through None and ``'unknown'``."""
    if not sha or sha == 'unknown':
        return None
    return sha[:SHORT_SHA_LEN]


def pi_versions(scene_dir: pathlib.Path) -> list[str]:
    """
    Return the distinct ``polyumi_version`` values across scene_dir's session metadata.

    Normally one entry: every session in a scene is recorded in one sitting off one
    deploy. More than one means the Pi was redeployed mid-scene, which is worth seeing
    rather than collapsing to "the" recording commit.

    Read with plain ``json`` rather than ``SessionMetadata.from_file``: this needs one
    field, and a metadata file that fails full dataclass validation should still be able
    to report the commit that wrote it.
    """
    seen: list[str] = []
    if not scene_dir.is_dir():
        return seen
    for session_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith('session_')):
        md_path = session_dir / 'metadata.json'
        if not md_path.is_file():
            continue
        try:
            version = json.loads(md_path.read_text()).get('polyumi_version')
        except (OSError, ValueError) as err:
            log.warning(f'Could not read polyumi_version from {md_path}: {err}')
            continue
        if version and version not in seen:
            seen.append(version)
    return seen


def scene_provenance(scene_dir: pathlib.Path) -> dict:
    """
    Return the recording + preprocessing commit provenance for scene_dir.

    ``pzarr_git_sha`` / ``pipeline_version`` / ``pzarr_created_at`` are None when no pzarr
    has been built. Per-step commits are not here — they come through
    ``pp_status.scene_pp_status``, which already walks the same attrs to build the step list.
    """
    import zarr

    from polyumi_ingest.pzarr.scene_files import SceneFiles

    out: dict = {
        'pi_versions': pi_versions(scene_dir),
        'pzarr_git_sha': None,
        'pipeline_version': None,
        'pzarr_created_at': None,
    }
    zarr_path = SceneFiles.resolve_zarr_path(scene_dir)
    if not zarr_path.exists():
        return out

    root = zarr.open_group(str(zarr_path), mode='r')
    out['pzarr_git_sha'] = root.attrs.get('git_sha')
    out['pipeline_version'] = root.attrs.get('pipeline_version')
    out['pzarr_created_at'] = root.attrs.get('created_at')
    return out
