"""
Per-session pzarr stream inspection for the detail pane (Phase 4).

Wraps ingest's own ``inspect_pzarr`` (the same summary ``pingest inspect-zarr``
prints), scoped down to just the one episode a selected session maps to, and
formatted the same way as that CLI's per-episode table — no new computation,
just reusing ingest's reader and adapting its display logic into a view model.
"""

from __future__ import annotations

import pathlib

import zarr

from polyumi_catalog.mcap_tools import resolve_episode_index

# (EpisodeInfo attribute name, display label) — same order/labels as `pingest inspect-zarr`.
_STREAMS = [
    ('finger', 'finger/frames'),
    ('finger_piezo', 'finger/finger_piezo'),
    ('finger_air', 'finger/finger_air'),
    ('gopro', 'gopro/frames'),
    ('gopro_accl', 'gopro/accl'),
    ('gopro_gyro', 'gopro/gyro'),
    ('gopro_gps', 'gopro/gps'),
    ('gopro_audio', 'gopro/audio'),
]


def _fmt_rate(freq_hz: float | None) -> str | None:
    """Format a sample rate the same way `pingest inspect-zarr` does."""
    if freq_hz is None:
        return None
    if freq_hz >= 1000:
        return f'{freq_hz / 1000:.1f} kHz'
    return f'{freq_hz:.2f} Hz'


def _fmt_ts_range(ts_range: tuple[float, float] | None) -> str | None:
    """Format a (start, end) timestamp range in seconds."""
    if ts_range is None:
        return None
    return f'{ts_range[0]:.3f} → {ts_range[1]:.3f} s'


def available_pose_sources(scene_dir: pathlib.Path, session_dirname: str) -> list[str] | None:
    """
    Return this session's ``eef.attrs['available_sources']`` (preprocessing step 5), or ``None``.

    ``None`` means "unknown" — no pzarr yet, no matching episode, or step 5 hasn't run for it —
    distinct from an empty list, which can't actually happen (step 5 skips episodes with no
    source at all rather than writing an empty ``eef`` group). Callers should treat ``None`` as
    "don't restrict the pose-source choice", not "no sources".
    """
    idx = resolve_episode_index(scene_dir, session_dirname)
    if idx is None:
        return None
    zarr_path = scene_dir / 'scene.zarr'
    ep = zarr.open_group(str(zarr_path / f'episode_{idx}'), mode='r')
    if 'eef' not in ep:
        return None
    available = ep['eef'].attrs.get('available_sources')  # type: ignore[union-attr]
    return list(available) if available else None


def session_pzarr_streams(scene_dir: pathlib.Path, session_dirname: str) -> dict | None:
    """
    Return this session's pzarr stream summary, or ``None`` if unavailable.

    Unavailable means: no pzarr built yet, or no episode group matches this session.
    Streams with no array present (e.g. ``gopro/frames``, dropped from pzarr in favor
    of on-demand mp4 decoding) are omitted, matching the CLI's own behavior.
    """
    idx = resolve_episode_index(scene_dir, session_dirname)
    if idx is None:
        return None

    from polyumi_ingest.pzarr import inspect_pzarr

    info = inspect_pzarr(scene_dir)
    ep = next((e for e in info.episodes if e.index == idx), None)
    if ep is None:
        return None

    streams = []
    for attr_name, label in _STREAMS:
        stream = getattr(ep, attr_name)
        if stream.shape is None:
            continue
        streams.append(
            {
                'label': label,
                'shape': stream.shape,
                'rate': _fmt_rate(stream.freq_hz),
                'timestamps': _fmt_ts_range(stream.ts_range),
            }
        )

    duration_s = None
    if ep.episode_start is not None and ep.episode_end is not None:
        duration_s = ep.episode_end - ep.episode_start

    return {
        'episode_index': ep.index,
        'episode_start': ep.episode_start,
        'episode_end': ep.episode_end,
        'duration_s': duration_s,
        'streams': streams,
    }
