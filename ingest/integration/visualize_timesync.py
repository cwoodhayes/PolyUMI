"""
Visualize time-sync alignment for a scene after preprocessing step 1 (pp).

Usage:
    uv run python ingest/integration/visualize_timesync.py recordings/scene_YYYY-MM-DD_...
"""

import argparse
import logging
import os
import pathlib
import signal
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import zarr
from matplotlib.axes import Axes
from polyumi_ingest.preproc.audio_align import GCCPHATAligner, PowerEnvAligner
from polyumi_ingest.preproc.time_sync import ChirpTimeSyncStep, TimeSyncStep
from polyumi_ingest.pzarr.scene_files import SceneFiles
from rich.logging import RichHandler


def _load(grp: zarr.Group, path: str) -> np.ndarray:
    return np.asarray(grp[path][()])  # type: ignore[index]


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def _slice_window(
    ts_arr: zarr.Array, data_arr: zarr.Array, t_start: float, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """Load only the samples within [t_start, t_end] from zarr arrays."""
    # Approximate index bounds using uniform spacing assumption (avoids loading full ts array).
    n = ts_arr.shape[0]
    t0 = float(ts_arr[0])
    t1 = float(ts_arr[-1])
    dt = (t1 - t0) / (n - 1) if n > 1 else 1.0
    i0 = max(0, int((t_start - t0) / dt) - 2)
    i1 = min(n, int((t_end - t0) / dt) + 2)
    ts = np.asarray(ts_arr[i0:i1])
    data = np.asarray(data_arr[i0:i1])
    mask = (ts >= t_start) & (ts <= t_end)
    return ts[mask], data[mask]


def _plot_episode(
    ep: zarr.Group,
    episode_key: str,
    axes: list[Axes],
    window_s: float = 8.0,
    unaligned: bool = True,
) -> None:
    ann = ep['annotations/time_sync'].attrs  # type: ignore[index]
    total_offset = float(ann['gopro_to_finger_offset_s'])  # type: ignore[arg-type]

    gopro_t0 = float(ep['timestamps/gopro_audio'][0])
    finger_align_t = gopro_t0 - total_offset

    # Window in absolute time for each stream.
    # In aligned mode the GoPro window is shifted back by total_offset so both
    # streams land at t=0 on a shared axis. In unaligned mode each stream is
    # windowed around its own natural start time so the clock offset is visible.
    finger_t_start = finger_align_t - 1.0
    finger_t_end = finger_align_t + window_s
    if unaligned:
        gopro_t_start = gopro_t0 - 1.0
        gopro_t_end = gopro_t0 + window_s
    else:
        gopro_t_start = finger_align_t - 1.0
        gopro_t_end = finger_align_t + window_s

    piezo_ts, piezo = _slice_window(
        ep['timestamps/finger_piezo'],
        ep['finger/finger_piezo'],
        finger_t_start,
        finger_t_end,  # type: ignore[arg-type]
    )
    air_ts, air = _slice_window(
        ep['timestamps/finger_air'],
        ep['finger/finger_air'],
        finger_t_start,
        finger_t_end,  # type: ignore[arg-type]
    )
    gopro_ts, gopro_raw = _slice_window(
        ep['timestamps/gopro_audio'],
        ep['gopro/audio'],
        gopro_t_start,
        gopro_t_end,  # type: ignore[arg-type]
    )
    gopro = _mono(gopro_raw)

    if 'finger_chirp_peak' in ann:
        finger_peak = float(ann['finger_chirp_peak'])  # type: ignore[arg-type]
        gopro_peak = float(ann['gopro_chirp_peak'])  # type: ignore[arg-type]
        peak_label = f'finger_peak={finger_peak:.3f}  gopro_peak={gopro_peak:.3f}'
    else:
        peak = float(ann['peak'])  # type: ignore[arg-type]
        peak_label = f'peak={peak:.3f}'

    # Shared time reference: finger stream always uses finger_align_t as origin.
    # In aligned mode GoPro also uses finger_align_t (= gopro_t0 - offset) so the
    # two chirps overlap. In unaligned mode GoPro uses gopro_t0, preserving the
    # raw clock offset between the devices.
    gopro_ref = gopro_t0 if unaligned else finger_align_t

    finger_chirp_x: float | None = None
    gopro_chirp_x: float | None = None
    if 'finger_chirp_onset_s' in ann:
        finger_chirp_x = float(ann['finger_chirp_onset_s']) - finger_align_t  # type: ignore[arg-type]
    if 'gopro_chirp_onset_s' in ann:
        gopro_chirp_x = float(ann['gopro_chirp_onset_s']) - gopro_ref  # type: ignore[arg-type]

    bar_label = f'alignment point  (offset={total_offset:+.4f}s, {peak_label})'
    mode_tag = 'unaligned' if unaligned else 'aligned'

    traces = [
        (axes[0], piezo_ts - finger_align_t, piezo, 'steelblue', 'finger piezo (RPi)', finger_chirp_x),
        (axes[1], air_ts - finger_align_t, air, 'steelblue', 'finger mic (RPi)', finger_chirp_x),
        (axes[2], gopro_ts - gopro_ref, gopro, 'darkorange', 'GoPro mic', gopro_chirp_x),
    ]
    for i, (ax, ts, data, color, ylabel, chirp_x) in enumerate(traces):
        ax: Axes  # type: ignore
        ax.plot(ts, data, linewidth=0.3, color=color, rasterized=True)
        if not unaligned:
            ax.axvline(0, color='red', linewidth=1.2, label=bar_label if i == 0 else None)
        if chirp_x is not None:
            ax.axvline(
                chirp_x,
                color='green',
                linewidth=3.0,
                linestyle='--',
                label=f'chirp onset ({chirp_x:.3f}s)' if i == 1 else None,
            )
        ax.yaxis.set_label_position('right')
        ax.tick_params(axis='x', labelsize=7, labelbottom=True)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_title(ylabel)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        ax.grid(True, axis='x', which='minor', linewidth=0.4, alpha=0.6)
        ax.grid(True, axis='x', which='major', linewidth=0.8, alpha=0.8)

    axes[1].legend(fontsize=12, loc='upper right')
    axes[0].set_title(f'{episode_key}  [{mode_tag}]', loc='left', fontsize=8, pad=2)
    if unaligned:
        axes[2].set_title('Time since start of session (for each system)', loc='right', fontsize=8, pad=2)
    else:
        axes[2].set_xlabel('time relative to finger alignment point (s)', fontsize=8)


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
        format='%(message)s',
        handlers=[RichHandler(show_time=True, show_level=True, show_path=False, rich_tracebacks=True)],
    )
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene', type=pathlib.Path, help='Scene directory or scene.zarr path.')
    parser.add_argument(
        '--unaligned',
        action='store_true',
        help='Show raw (unaligned) GoPro audio without applying the time offset.',
    )
    args = parser.parse_args()

    scene_zarr = SceneFiles.resolve_zarr_path(args.scene)
    if not scene_zarr.exists():
        print(f'error: no scene.zarr found at {args.scene}', file=sys.stderr)
        sys.exit(1)

    # step = TimeSyncStep(aligner=GCCPHATAligner(alpha=0.0), max_lag_s=2.0, trim_start_s=0.8)
    # step = TimeSyncStep()
    # step = TimeSyncStep(aligner=PowerEnvAligner(power=1.2), trim_start_s=0.7, max_lag_s=1.5)
    # step = ChirpTimeSyncStep()
    # print("Running time sync step...")
    # scene_zarr = step.run(args.scene, copy=True, force=True)
    print('Time sync step completed. Visualizing results...')
    scene_zarr = SceneFiles.resolve_zarr_path(args.scene)
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
        _plot_episode(ep, episode_key, list(axes), unaligned=args.unaligned)  # type: ignore[arg-type]
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
