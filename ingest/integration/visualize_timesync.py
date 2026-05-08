"""
Visualize time-sync alignment for a scene after preprocessing step 1 (pp).

Usage:
    uv run python ingest/integration/visualize_timesync.py recordings/scene_YYYY-MM-DD_...
"""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.axes import Axes
from polyumi_ingest.pzarr.scene_files import SceneFiles


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

    # Red bar marks "GoPro t0" in each stream's native timestamp domain.
    # total_offset ≈ gopro_ts[0] - finger_ts[0], so GoPro t0 in finger time = gopro_ts[0] - total_offset.
    gopro_t0 = float(gopro_ts[0])
    finger_align_t = gopro_t0 - total_offset
    bar_label = f'GoPro t0  (offset={total_offset:+.4f}s, peak={peak:.3f})'

    traces = [
        (axes[0], piezo_ts, piezo, 'steelblue', 'finger piezo'),
        (axes[1], air_ts, air, 'steelblue', 'finger air'),
        (axes[2], gopro_ts, gopro, 'darkorange', 'GoPro audio (mono)'),
    ]
    for i, (ax, ts, data, color, ylabel) in enumerate(traces):
        ax: Axes  # type: ignore
        ax.plot(ts, data, linewidth=0.3, color=color, rasterized=True)
        bar_t = gopro_t0 if i == 2 else finger_align_t
        ax.axvline(bar_t, color='red', linewidth=1.2, label=bar_label if i == 0 else None)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.yaxis.set_label_position('right')
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_title(ylabel)
        if i < 2:
            ax.set_xlabel('')

    axes[0].legend(fontsize=7, loc='upper right')
    axes[0].set_title(episode_key, loc='left', fontsize=8, pad=2)
    axes[2].set_xlabel('UTC time (s)', fontsize=8)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene', type=pathlib.Path, help='Scene directory or scene.zarr path.')
    args = parser.parse_args()

    scene_zarr = SceneFiles.resolve_zarr_path(args.scene)
    if not scene_zarr.exists():
        print(f'error: no scene.zarr found at {args.scene}', file=sys.stderr)
        sys.exit(1)

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

        fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=False)
        fig.suptitle(f'{scene_zarr.parent.name}  /  {episode_key}', fontsize=9)
        _plot_episode(ep, episode_key, list(axes))  # type: ignore[arg-type]
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    plt.show()


if __name__ == '__main__':
    main()
