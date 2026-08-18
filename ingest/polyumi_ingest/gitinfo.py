"""
Provenance helper: which commit of this repo produced a given artifact.

Stamped into the pzarr at build time (``git_sha`` on the root group) and again per
preprocessing step (``preprocessing_step_versions``), so a store can be traced back to
the code that wrote it. That matters when the pipeline's behaviour changes under a
corpus that is only partly reprocessed — e.g. a SLAM config migration, where the stored
poses mean different things depending on which commit produced them.
"""

from __future__ import annotations

import pathlib
import subprocess

#: Recorded when the repo can't be identified — kept as a string rather than None so
#: consumers never have to special-case the type.
UNKNOWN_SHA = 'unknown'

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _resolve_git_sha() -> str:
    """
    Shell out for the repo's HEAD commit, or return ``'unknown'``.

    Resolved against this file's own location rather than the process cwd: the catalog
    server and the ingest CLI are routinely run from elsewhere, and a cwd-relative
    lookup would silently stamp whatever unrelated repo the shell happened to be in.

    Timed out rather than left to block: this now runs at import, so a git that hangs
    (a stale lock, an unresponsive filesystem) would take the whole process down with
    it instead of only the first caller. TimeoutExpired is a SubprocessError, so the
    handler below already covers it.
    """
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_SHA
    return out.strip() or UNKNOWN_SHA


#: Captured at **import**, which is the closest thing to "the commit whose code is
#: actually running" that we can cheaply observe.
#:
#: This used to be resolved lazily on first call, which is subtly wrong for any process
#: that outlives a commit. The catalog server can trigger preprocessing from the UI and
#: stays up for days; on 2026-08-17 one had been running since before a batch of commits
#: landed, so it executed the *old* eef-pose step while `git rev-parse` — run fresh at
#: marking time — reported the *new* HEAD. Scene 30ed ended up stamped with a commit whose
#: code had never touched it, and the missing `max_pose_jump_m` attr it should have written
#: looked, from the provenance alone, impossible.
#:
#: Import time is not a perfect proxy — an editable checkout can still be edited underneath
#: a running process without any commit at all — but it moves the stamp from "whenever
#: someone asked" to "when this code was loaded", which is the question it is meant to
#: answer.
_LOADED_GIT_SHA = _resolve_git_sha()


def git_sha() -> str:
    """Return the commit this process's code was loaded from, or ``'unknown'``."""
    return _LOADED_GIT_SHA
