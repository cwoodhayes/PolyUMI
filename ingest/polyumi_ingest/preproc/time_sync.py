"""Audio time synchronization preprocessing step."""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import zarr

from polyumi_ingest.preproc.step_base import (
    PreprocessingStep,
    _mark_preprocessing_step,
    _write_scalar,
    register_preprocessing_step,
)

log = logging.getLogger(__name__)


def _arr(grp: zarr.Group, path: str) -> zarr.Array:
    """Return a typed zarr.Array from a group by path."""
    return grp[path]  # type: ignore[return-value]


def _infer_sample_rate(ts: np.ndarray) -> float:
    if len(ts) < 2:
        raise ValueError('Need at least two timestamps to infer sample rate')
    diffs = np.diff(ts.astype(np.float64))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError('Could not infer sample rate from timestamps')
    return float(1.0 / np.median(diffs))


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1).astype(np.float32, copy=False)


def _resample_to_grid(ts: np.ndarray, values: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    return np.interp(target_ts, ts, values).astype(np.float32)


def _gcc_phat(sig: np.ndarray, refsig: np.ndarray) -> tuple[int, float]:
    """Estimate the sample lag between sig and refsig using GCC-PHAT."""
    sig = np.asarray(sig, dtype=np.float64)
    refsig = np.asarray(refsig, dtype=np.float64)
    n = len(sig) + len(refsig)
    nfft = 1 << (n - 1).bit_length()
    sig_fft = np.fft.rfft(sig, n=nfft)
    ref_fft = np.fft.rfft(refsig, n=nfft)
    cross_power = sig_fft * np.conj(ref_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    cc = np.fft.irfft(cross_power, n=nfft)
    max_shift = nfft // 2
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shifts = np.arange(-max_shift, max_shift + 1)
    best_index = int(np.argmax(cc))
    return int(shifts[best_index]), float(cc[best_index])


@register_preprocessing_step(step_number=1, step_name='time-sync')
class TimeSyncStep(PreprocessingStep):
    """Estimate the offset between finger air audio and GoPro audio."""

    def run_step(self, scene_zarr: pathlib.Path) -> None:
        """Read the audio streams from scene_zarr and write the estimated offset."""
        root = zarr.open_group(str(scene_zarr), mode='a')
        episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
        if not episodes:
            raise RuntimeError(f'No episodes found in {scene_zarr}')

        for episode_key in episodes:
            ep = root.require_group(episode_key)
            finger_ts = np.asarray(_arr(ep, 'timestamps/finger_air')[:], dtype=np.float64)
            gopro_ts = np.asarray(_arr(ep, 'timestamps/gopro_audio')[:], dtype=np.float64)
            finger_audio = _mono_audio(np.asarray(_arr(ep, 'finger/finger_air')[:]))
            gopro_audio = _mono_audio(np.asarray(_arr(ep, 'gopro/audio')[:]))

            overlap_start = max(float(finger_ts[0]), float(gopro_ts[0]))
            overlap_end = min(float(finger_ts[-1]), float(gopro_ts[-1]))
            if overlap_end <= overlap_start:
                raise RuntimeError(f'No audio overlap in {episode_key}')

            finger_sr = _infer_sample_rate(finger_ts)
            gopro_sr = _infer_sample_rate(gopro_ts)
            target_sr = float(min(8_000, finger_sr, gopro_sr))
            target_dt = 1.0 / target_sr
            target_ts = np.arange(overlap_start, overlap_end, target_dt, dtype=np.float64)
            if len(target_ts) < 32:
                raise RuntimeError(f'Audio overlap too short for time sync in {episode_key}')

            finger_overlap = _resample_to_grid(finger_ts, finger_audio, target_ts)
            gopro_overlap = _resample_to_grid(gopro_ts, gopro_audio, target_ts)

            finger_overlap -= finger_overlap.mean()
            gopro_overlap -= gopro_overlap.mean()
            finger_scale = float(np.std(finger_overlap)) or 1.0
            gopro_scale = float(np.std(gopro_overlap)) or 1.0
            finger_overlap /= finger_scale
            gopro_overlap /= gopro_scale

            lag_samples, peak = _gcc_phat(gopro_overlap, finger_overlap)
            residual_offset_s = lag_samples / target_sr
            nominal_offset_s = float(gopro_ts[0] - finger_ts[0])
            total_offset_s = nominal_offset_s + residual_offset_s

            step_group = ep.require_group('annotations').require_group('time_sync')
            _write_scalar(step_group, 'gopro_audio_to_finger_air_offset_s', total_offset_s)
            _write_scalar(step_group, 'nominal_start_offset_s', nominal_offset_s)
            _write_scalar(step_group, 'residual_offset_s', residual_offset_s)
            _write_scalar(step_group, 'lag_samples', lag_samples)
            _write_scalar(step_group, 'peak', peak)
            _write_scalar(step_group, 'target_sample_rate_hz', int(round(target_sr)))

            log.info(
                f'{episode_key}: offset={total_offset_s:.6f}s '
                f'(nominal={nominal_offset_s:.6f}s, residual={residual_offset_s:.6f}s, peak={peak:.4f})'
            )

        _mark_preprocessing_step(root, self.step_number)
