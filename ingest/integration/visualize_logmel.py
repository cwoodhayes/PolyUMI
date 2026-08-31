"""
Visualize the diagnostic contact-mic log-mel spectrogram from preprocessing step 6 (pp).

Nothing trains on this array — it answers "did the piezo hear the contact, and in which band".
See polyumi_ingest.preproc.logmel.

Usage:
    uv run python ingest/integration/visualize_logmel.py recordings/scene_YYYY-MM-DD_...
"""

import argparse
import pathlib
import signal
import sys

import matplotlib.pyplot as plt
import numpy as np
import zarr
from polyumi_ingest.preproc.logmel import display_range, hz_to_mel, mel_to_hz
from polyumi_ingest.pzarr.scene_files import SceneFiles


def _mel_bin_hz(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Centre frequency of each mel bin, for labelling the y axis in Hz instead of bin index."""
    return mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))[1:-1]


def _plot_episode(ep: zarr.Group, episode_key: str, axes: list) -> None:
    """Draw the piezo waveform over its spectrogram, on a shared time axis."""
    grp = ep['annotations/contact_audio']  # type: ignore[index]
    logmel = np.asarray(grp['logmel'][:])  # type: ignore[index]
    logmel_ts = np.asarray(grp['logmel_timestamps'][:])  # type: ignore[index]
    piezo = np.asarray(ep['finger/finger_piezo'][:])  # type: ignore[index]
    piezo_ts = np.asarray(ep['timestamps/finger_piezo'][:])  # type: ignore[index]

    t0 = float(piezo_ts[0])
    attrs = grp.attrs
    n_mels = int(attrs['logmel_n_mels'])
    fmin, fmax = float(attrs['logmel_fmin']), float(attrs['logmel_fmax'])

    axes[0].plot(piezo_ts - t0, piezo, linewidth=0.3, color='steelblue', rasterized=True)
    axes[0].set_ylabel('amplitude', fontsize=8)
    axes[0].set_title(
        f'{episode_key}  —  rms {float(attrs["rms"]):.4f}, coverage {float(attrs["coverage"]):.3f}, {len(logmel)} hops',
        loc='left',
        fontsize=8,
    )

    vmin, vmax = display_range(logmel)
    # imshow rows run top-down and logmel is (n_hops, n_mels) with bin 0 lowest, so transpose
    # and let origin='lower' put low frequencies at the bottom.
    axes[1].imshow(
        logmel.T,
        origin='lower',
        aspect='auto',
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        extent=(float(logmel_ts[0]) - t0, float(logmel_ts[-1]) - t0, 0, n_mels),
    )
    bin_hz = _mel_bin_hz(n_mels, fmin, fmax)
    ticks = np.linspace(0, n_mels - 1, 6).astype(int)
    axes[1].set_yticks(ticks + 0.5)
    axes[1].set_yticklabels([f'{bin_hz[i]:.0f}' for i in ticks], fontsize=6)
    axes[1].set_ylabel('mel bin centre (Hz)', fontsize=8)
    axes[1].set_xlabel('time since start of piezo recording (s)', fontsize=8)

    for ax in axes:
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(True, axis='x', linewidth=0.4, alpha=0.5)


def main() -> None:
    """Entry point."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene', type=pathlib.Path, help='Scene directory or scene.zarr path.')
    parser.add_argument('--episode', type=int, default=None, help='Only plot this episode index.')
    args = parser.parse_args()

    scene_zarr = SceneFiles.resolve_zarr_path(args.scene)
    if not scene_zarr.exists():
        print(f'error: no scene.zarr found at {args.scene}', file=sys.stderr)
        sys.exit(1)

    root = zarr.open_group(str(scene_zarr), mode='r')
    episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
    if args.episode is not None:
        episodes = [k for k in episodes if k == f'episode_{args.episode}']

    plotted = 0
    for episode_key in episodes:
        ep = root[episode_key]  # type: ignore[index]
        if 'annotations/contact_audio/logmel' not in ep:
            print(f'skipping {episode_key}: no contact_audio annotation (run pingest pp 6 first)')
            continue
        if ep['annotations/contact_audio/logmel'].shape[0] == 0:  # type: ignore[index,union-attr]
            print(f'skipping {episode_key}: spectrogram is empty (episode shorter than one FFT frame)')
            continue

        fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True, height_ratios=(1, 3))
        fig.suptitle(f'{scene_zarr.parent.name}  /  {episode_key}', fontsize=9)
        _plot_episode(ep, episode_key, list(axes))  # type: ignore[arg-type]
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        plotted += 1

    if not plotted:
        print('error: nothing to plot', file=sys.stderr)
        sys.exit(1)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
