"""Audio time synchronization preprocessing step."""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import zarr
from polyumi_pi import sync_chirp

from polyumi_ingest.preproc.audio_align import AudioAligner, ChirpAligner, GCCPHATAligner
from polyumi_ingest.preproc.step_base import (
    PreprocessingStep,
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


class TimeSyncStep(PreprocessingStep):
    """Estimate the offset between finger air audio and GoPro audio."""

    step_number = 99
    step_name = 'time-sync-legacy'

    def __init__(
        self,
        max_lag_s: float = 2.0,
        aligner: AudioAligner | None = None,
        trim_start_s: float = 0.5,
    ) -> None:
        """
        Initialize the time-sync step.

        Parameters
        ----------
        max_lag_s:
            Search window passed to the aligner (±seconds).
        aligner:
            AudioAligner instance to use. Defaults to GCCPHATAligner(alpha=0.0)
            (standard cross-correlation, no spectral whitening), which outperforms
            full PHAT on piezo signals with dominant transients.
        trim_start_s:
            Seconds to discard from the start of each signal before alignment,
            to avoid the hardware turn-on transient on the piezo and air mics.

        """
        self.max_lag_s = max_lag_s
        self.aligner = aligner if aligner is not None else GCCPHATAligner(0.0)
        self.trim_start_s = trim_start_s
        self.finger_mic_name = 'finger_piezo'

    def run_step(self, scene_zarr: pathlib.Path, force: bool = False) -> None:
        """Read the audio streams from scene_zarr and write the estimated offset."""
        root = zarr.open_group(str(scene_zarr), mode='a')
        episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
        if not episodes:
            raise RuntimeError(f'No episodes found in {scene_zarr}')

        for episode_key in episodes:
            ep = root.require_group(episode_key)
            finger_ts = np.asarray(_arr(ep, f'timestamps/{self.finger_mic_name}')[:], dtype=np.float64)
            gopro_ts = np.asarray(_arr(ep, 'timestamps/gopro_audio')[:], dtype=np.float64)
            finger_audio = _mono_audio(np.asarray(_arr(ep, f'finger/{self.finger_mic_name}')[:]))
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

            trim_samples = int(self.trim_start_s * target_sr)
            finger_overlap = finger_overlap[trim_samples:]
            gopro_overlap = gopro_overlap[trim_samples:]
            if len(finger_overlap) < 32:
                raise RuntimeError(f'Audio overlap too short after trim in {episode_key}')

            # gopro always starts a little bit ahead, so we restrict the search range to [0, +max_lag_s]
            max_lag_samples = (0, int(target_sr * self.max_lag_s))
            lag_samples, peak = self.aligner.estimate_lag(
                gopro_overlap, finger_overlap, max_lag_samples=max_lag_samples
            )
            residual_offset_s = lag_samples / target_sr
            nominal_offset_s = float(gopro_ts[0] - finger_ts[0])
            total_offset_s = nominal_offset_s + residual_offset_s

            step_group = ep.require_group('annotations').require_group('time_sync')
            _write_scalar(step_group, 'gopro_to_finger_offset_s', total_offset_s)
            _write_scalar(step_group, 'nominal_start_offset_s', nominal_offset_s)
            _write_scalar(step_group, 'residual_offset_s', residual_offset_s)
            _write_scalar(step_group, 'lag_samples', lag_samples)
            _write_scalar(step_group, 'peak', peak)
            _write_scalar(step_group, 'target_sample_rate_hz', int(round(target_sr)))

            log.info(
                f'{episode_key}: offset={total_offset_s:.6f}s '
                f'(nominal={nominal_offset_s:.6f}s, residual={residual_offset_s:.6f}s, peak={peak:.4f})'
                f' with aligner={self.aligner.__class__.__name__}'
            )


@register_preprocessing_step(step_number=1, step_name='chirp-time-sync')
class ChirpTimeSyncStep(PreprocessingStep):
    """
    Estimate the GoPro-to-finger offset by matched-filtering a known sync chirp.

    Both the finger air mic and GoPro audio are cross-correlated against the
    reference chirp (generated by ``polyumi_pi.sync_chirp``). The difference
    in detected onset times gives the inter-device clock offset.

    Convention: ``gopro_to_finger_offset_s = t_chirp_gopro - t_chirp_finger``.
    To put a GoPro timestamp on the finger timeline, subtract this value.
    This matches the sign used by the MCAP exporter.
    """

    def __init__(self, search_radius_s: float = 3.0) -> None:
        """
        Initialize.

        Parameters
        ----------
        search_radius_s:
            Half-width of the search window around the expected chirp play time.
            Only used when ``annotations/sync_chirp_play_time_s`` is present in
            the zarr. If the annotation is absent the full recording is searched.

        """
        self.search_radius_s = search_radius_s
        self._aligner = ChirpAligner()

    def run_step(self, scene_zarr: pathlib.Path, force: bool = False) -> None:
        """Detect chirp onsets in both recordings and write the estimated offset."""
        root = zarr.open_group(str(scene_zarr), mode='a')
        episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
        if not episodes:
            raise RuntimeError(f'No episodes found in {scene_zarr}')

        for episode_key in episodes:
            ep = root.require_group(episode_key)

            finger_air = _mono_audio(np.asarray(_arr(ep, 'finger/finger_air')[:]))
            finger_ts = np.asarray(_arr(ep, 'timestamps/finger_air')[:], dtype=np.float64)
            finger_sr = int(round(_infer_sample_rate(finger_ts)))

            gopro_audio = _mono_audio(np.asarray(_arr(ep, 'gopro/audio')[:]))
            gopro_ts = np.asarray(_arr(ep, 'timestamps/gopro_audio')[:], dtype=np.float64)
            gopro_sr = int(round(_infer_sample_rate(gopro_ts)))

            chirp_play_time_s: float | None = None
            if 'annotations' in ep and 'sync_chirp_play_time_s' in ep['annotations'].attrs:
                chirp_play_time_s = float(ep['annotations'].attrs['sync_chirp_play_time_s'])  # type: ignore[arg-type]

            ref_finger = sync_chirp.generate(finger_sr)
            ref_gopro = sync_chirp.generate(gopro_sr)

            if chirp_play_time_s is not None:
                radius_finger = int(self.search_radius_s * finger_sr)
                center_finger = int((chirp_play_time_s - finger_ts[0]) * finger_sr)
                window_finger = (center_finger - radius_finger, center_finger + radius_finger)

                radius_gopro = int(self.search_radius_s * gopro_sr)
                center_gopro = int((chirp_play_time_s - gopro_ts[0]) * gopro_sr)
                window_gopro = (center_gopro - radius_gopro, center_gopro + radius_gopro)
            else:
                window_finger = None
                window_gopro = None

            onset_finger, peak_finger = self._aligner.estimate_lag(
                finger_air, ref_finger, max_lag_samples=window_finger
            )
            onset_gopro, peak_gopro = self._aligner.estimate_lag(
                gopro_audio, ref_gopro, max_lag_samples=window_gopro
            )

            t_chirp_finger = finger_ts[0] + onset_finger / finger_sr
            t_chirp_gopro = gopro_ts[0] + onset_gopro / gopro_sr
            gopro_to_finger_offset_s = t_chirp_gopro - t_chirp_finger

            step_group = ep.require_group('annotations').require_group('time_sync')
            _write_scalar(step_group, 'gopro_to_finger_offset_s', gopro_to_finger_offset_s)
            _write_scalar(step_group, 'finger_chirp_onset_s', t_chirp_finger)
            _write_scalar(step_group, 'gopro_chirp_onset_s', t_chirp_gopro)
            _write_scalar(step_group, 'finger_chirp_peak', peak_finger)
            _write_scalar(step_group, 'gopro_chirp_peak', peak_gopro)

            log.info(
                f'{episode_key}: offset={gopro_to_finger_offset_s:.6f}s '
                f'(finger_onset={t_chirp_finger:.3f}s, gopro_onset={t_chirp_gopro:.3f}s, '
                f'finger_peak={peak_finger:.4f}, gopro_peak={peak_gopro:.4f})'
            )
