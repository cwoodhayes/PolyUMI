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

    Timed out because this runs at import: a git that hangs (a stale lock, an unresponsive
    filesystem) would otherwise take the whole process down with it. TimeoutExpired is a
    SubprocessError, so the handler below already covers it.
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


#: Captured at **import**, not on first call: the catalog server can trigger preprocessing
#: from the UI and stays up for days, so a lazily resolved sha reports whatever HEAD is at
#: marking time — which may be several commits past the code actually doing the work.
#:
#: Not a perfect proxy (an editable checkout can be edited underneath a running process with
#: no commit at all), but it answers the question the stamp is asked: which code ran.
_LOADED_GIT_SHA = _resolve_git_sha()


def git_sha() -> str:
    """Return the commit this process's code was loaded from, or ``'unknown'``."""
    return _LOADED_GIT_SHA
