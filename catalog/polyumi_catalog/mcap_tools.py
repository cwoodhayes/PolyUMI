"""
Per-episode MCAP export + local Foxglove launch (Phase 2.5).

MCAP export reuses ingest's existing exporter (``polyumi_ingest.export.mcap``)
rather than reimplementing anything, per the "ingest owns preprocessing/export"
decision. The one bit of glue this module adds is
resolving a catalog ``Session`` to its pzarr *episode index*: pzarr keys episodes
by position (``episode_0``, ``episode_1``, ...) rather than by session_id, so we
recover the mapping via each episode group's ``session_dir`` attribute, which
``build_pzarr`` stamps from the source session directory name.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import zarr


class McapError(ValueError):
    """An MCAP export or Foxglove-launch operation could not be completed."""


def pzarr_exists(scene_dir: pathlib.Path) -> bool:
    """Report whether ``scene_dir`` has a built pzarr store (``scene.zarr``)."""
    return (scene_dir / 'scene.zarr').is_dir()


def resolve_episode_index(scene_dir: pathlib.Path, session_dirname: str) -> int | None:
    """
    Return the pzarr episode index for the session directory named ``session_dirname``.

    Returns ``None`` if pzarr doesn't exist yet, or no episode group matches.
    """
    zarr_path = scene_dir / 'scene.zarr'
    if not zarr_path.is_dir():
        return None
    root = zarr.open_group(str(zarr_path), mode='r')
    n_episodes = int(root.attrs.get('n_episodes', 0))
    for i in range(n_episodes):
        ep_key = f'episode_{i}'
        if ep_key not in root:
            continue
        if root[ep_key].attrs.get('session_dir') == session_dirname:
            return i
    return None


def mcap_path_for_session(scene_dir: pathlib.Path, session_dirname: str) -> pathlib.Path | None:
    """Return this session's exported ``.mcap`` path if it already exists, else ``None``."""
    idx = resolve_episode_index(scene_dir, session_dirname)
    if idx is None:
        return None
    path = scene_dir / f'episode_{idx}.mcap'
    return path if path.is_file() else None


def export_session_to_mcap(scene_dir: pathlib.Path, session_dirname: str) -> pathlib.Path:
    """Export this session's pzarr episode to MCAP, returning the written path."""
    from polyumi_ingest.export.mcap import export_scene_to_mcap

    idx = resolve_episode_index(scene_dir, session_dirname)
    if idx is None:
        raise McapError(f'No pzarr episode found for {session_dirname!r} (build pzarr first with `pingest pp`).')
    written = export_scene_to_mcap(scene_dir, episode=idx)
    if not written:
        raise McapError(f'Export produced no output for episode {idx}.')
    return written[0]


def open_in_foxglove(mcap_path: pathlib.Path) -> None:
    """Launch the local Foxglove desktop app on ``mcap_path``."""
    if not mcap_path.is_file():
        raise McapError(f'No such file: {mcap_path}')
    binary = shutil.which('foxglove-studio')
    if binary is None:
        raise McapError('foxglove-studio is not installed (or not on PATH) on this machine.')
    subprocess.Popen([binary, str(mcap_path)], start_new_session=True)
