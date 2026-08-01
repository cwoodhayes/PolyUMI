"""
Provenance helper: which commit of this repo produced a given artifact.

Stamped into the pzarr at build time (``git_sha`` on the root group) and again per
preprocessing step (``preprocessing_step_versions``), so a store can be traced back to
the code that wrote it. That matters when the pipeline's behaviour changes under a
corpus that is only partly reprocessed — e.g. a SLAM config migration, where the stored
poses mean different things depending on which commit produced them.
"""

from __future__ import annotations

import functools
import pathlib
import subprocess

#: Recorded when the repo can't be identified — kept as a string rather than None so
#: consumers never have to special-case the type.
UNKNOWN_SHA = 'unknown'

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


@functools.lru_cache(maxsize=1)
def git_sha() -> str:
    """
    Return the current HEAD commit of the polyumi repo, or ``'unknown'``.

    Resolved against this file's own location rather than the process cwd: the catalog
    server and the ingest CLI are routinely run from elsewhere, and a cwd-relative
    lookup would silently stamp whatever unrelated repo the shell happened to be in.

    Cached — the answer can't change within a process, and this is called once per
    preprocessing step.
    """
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_SHA
    return out.strip() or UNKNOWN_SHA
