"""
Visualize time-sync alignment for a scene after preprocessing step 1 (pp).

Usage:
    uv run python ingest/integration/visualize_timesync.py recordings/scene_YYYY-MM-DD_...
"""

import argparse
import logging
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.axes import Axes
from polyumi_ingest.preproc.audio_align import GCCPHATAligner
from polyumi_ingest.preproc.time_sync import TimeSyncStep
from polyumi_ingest.pzarr.scene_files import SceneFiles
from rich.logging import RichHandler


def _load(grp: zarr.Group, path: str) -> np.ndarray:
    return np.asarray(grp[path][()])  # type: ignore[index]


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def _plot_episode(ep: zarr.Group, episode_key: str, axes: list[Axes]) -> None:
    piezo_ts = _load(ep, 'timestamps/finger_piezo')
    air_ts = _load(ep, 'timestamps/finger_air')
    gopro_ts = _load(ep, 'timestamps/gopro_audio')

    piezo = _load(ep, 'finger/finger_piezo')
    air = _load(ep, 'finger/finger_air')
    gopro = _mono(_load(ep, 'gopro/audio'))

    total_offset = float(_load(ep, 'annotations/time_sync/gopro_audio_to_finger_air_offset_s'))
    peak = float(_load(ep, 'annotations/time_sync/peak'))

    # Shift each stream so that the alignment point (GoPro t0) lands at t=0 on a shared axis.
    # total_offset ≈ gopro_ts[0] - finger_ts[0], so GoPro t0 in finger time = gopro_ts[0] - total_offset.
    gopro_t0 = float(gopro_ts[0])
    finger_align_t = gopro_t0 - total_offset

    bar_label = f'alignment point  (offset={total_offset:+.4f}s, peak={peak:.3f})'

    traces = [
        (axes[0], piezo_ts - finger_align_t, piezo, 'steelblue', 'finger piezo'),
        (axes[1], air_ts - finger_align_t, air, 'steelblue', 'finger air'),
        (axes[2], gopro_ts - gopro_t0, gopro, 'darkorange', 'GoPro audio (mono)'),
    ]
    for i, (ax, ts, data, color, ylabel) in enumerate(traces):
        ax: Axes  # type: ignore
        ax.plot(ts, data, linewidth=0.3, color=color, rasterized=True)
        ax.axvline(0, color='red', linewidth=1.2, label=bar_label if i == 0 else None)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.yaxis.set_label_position('right')
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_title(ylabel)

    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].set_title(episode_key, loc='left', fontsize=8, pad=2)
    axes[2].set_xlabel('time relative to alignment point (s)', fontsize=8)


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
        format='%(message)s',
        handlers=[RichHandler(show_time=True, show_level=True, show_path=False, rich_tracebacks=True)],
    )
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene', type=pathlib.Path, help='Scene directory or scene.zarr path.')
    args = parser.parse_args()

    scene_zarr = SceneFiles.resolve_zarr_path(args.scene)
    if not scene_zarr.exists():
        print(f'error: no scene.zarr found at {args.scene}', file=sys.stderr)
        sys.exit(1)

    # step = TimeSyncStep(aligner=GCCPHATAligner(alpha=0.0))
    step = TimeSyncStep()
    scene_zarr = step.run(args.scene, copy=True, force=True)
    root = zarr.open_group(str(scene_zarr), mode='r')
    episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
    if not episodes:
        print('error: no episodes found', file=sys.stderr)
        sys.exit(1)

    for episode_key in episodes:
        ep = root[episode_key]  # type: ignore[index]
        if 'annotations/time_sync' not in ep:
            print(f'skipping {episode_key}: no time_sync annotation (run pingest pp 1 first)')
            continue

        fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True)
        fig.suptitle(f'{scene_zarr.parent.name}  /  {episode_key}', fontsize=9)
        _plot_episode(ep, episode_key, list(axes))  # type: ignore[arg-type]
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
