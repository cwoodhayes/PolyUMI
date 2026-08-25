"""Contact-mic preprocessing step: piezo audio onto the GoPro frame grid, plus a diagnostic mel."""

from __future__ import annotations

import logging

import numpy as np
from numcodecs import Blosc

from polyumi_ingest.config import load_contact_audio_config
from polyumi_ingest.episode_status import Episode, SceneContext
from polyumi_ingest.preproc.logmel import log_mel_spectrogram
from polyumi_ingest.preproc.step_base import PreprocessingStep, register_preprocessing_step
from polyumi_ingest.pzarr.store import arr
from polyumi_ingest.timebase import gopro_ts_in_finger_clock

log = logging.getLogger(__name__)

_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

#: Tolerance on the sample rate inferred from timestamps before the step refuses to run, as a
#: fraction of the configured rate. Timestamps are synthesised as ``start + arange(n)/sr`` so the
#: inferred value is near-exact; this only absorbs float noise, not a genuinely different device.
_SAMPLE_RATE_TOLERANCE = 0.01


@register_preprocessing_step(step_number=6, step_name='contact-audio')
class ContactAudioStep(PreprocessingStep):
    """
    Slice the piezo contact mic into per-GoPro-frame blocks, and draw a diagnostic spectrogram.

    The blocks are the training product: ``pingest export --type polyumi`` concatenates the ``stride``
    of them belonging to each exported step into that step's ``mic_0`` row. Slicing here rather
    than in the exporter is what keeps the two clocks' reconciliation in one place — the GoPro
    and the Pi stamp against unrelated epochs, and step 1's chirp offset is the only bridge.

    Blocks are anchored per frame by ``searchsorted`` on that frame's own timestamp, never by
    multiplying a rate: at 16 kHz and 59.94 fps there are 266.93 samples per frame, so any fixed
    multiply would walk off the audio over an episode. Their fixed *width* is configured a little
    wider than the largest integer spacing, which is what guarantees consecutive blocks abut or
    overlap and never leave a hole — the property the exporter's contiguity depends on.

    The log-mel spectrogram written alongside is **diagnostic only**. Nothing trains on it; the
    policy's spectrogram is computed in the training container from the exported waveform, after
    augmentation that only exists in the waveform domain. See ``docs/maniwav-audio-policy.md``.
    """

    #: The visuomotor export reads none of this step's output, so requiring it would strand every
    #: scene preprocessed before this step existed. ``--type polyumi`` demands it via its
    #: modality's ``required_steps`` instead.
    required_for_export = False

    def prepare_scene(self, scene: SceneContext) -> None:
        """Load the block geometry and spectrogram parameters shared by every episode."""
        cfg = load_contact_audio_config()
        blocks = cfg['blocks']
        self.sample_rate_hz = int(blocks['sample_rate_hz'])
        self.block_width = int(blocks['samples_per_gopro_frame'])
        self.block_alignment = str(blocks['block_alignment'])
        if self.block_alignment not in ('causal', 'forward'):
            raise ValueError(
                f'block_alignment must be "causal" or "forward", got {self.block_alignment!r} '
                f'(config/contact_audio.yaml)'
            )
        self.logmel_cfg = dict(cfg['logmel'])

    def process_episode(self, scene: SceneContext, episode: Episode) -> None:
        """Write one episode's per-frame audio blocks and its diagnostic spectrogram."""
        ep, episode_key = episode.group, episode.key

        if episode.is_mapping:
            log.info(f'{episode_key}: MAPPING session, skipping contact audio.')
            return
        for path in ('timestamps/gopro', 'finger/finger_piezo', 'timestamps/finger_piezo'):
            if path not in ep:
                log.warning(f'{episode_key}: no {path}; skipping contact audio.')
                return

        piezo = np.asarray(arr(ep, 'finger/finger_piezo')[:], dtype=np.float32)
        piezo_ts = np.asarray(arr(ep, 'timestamps/finger_piezo')[:], dtype=np.float64)
        if len(piezo) < 2:
            log.warning(f'{episode_key}: only {len(piezo)} piezo sample(s); skipping contact audio.')
            return
        self._check_sample_rate(episode_key, piezo_ts)

        # Sample-exact slicing against finger audio, so an unshifted grid is not a degraded
        # result but a wrong one — hence require_offset.
        gopro_ts_finger = gopro_ts_in_finger_clock(ep, require_offset=True)
        n_frames = len(gopro_ts_finger)
        starts = np.searchsorted(piezo_ts, gopro_ts_finger, side='left').astype(np.int64)

        blocks, n_zero_filled = self._gather_blocks(piezo, starts)
        spacing = np.diff(starts)
        max_spacing = int(spacing.max()) if len(spacing) else 0
        n_gaps = int((spacing > self.block_width).sum())
        if n_gaps:
            log.warning(
                f'{episode_key}: {n_gaps} frame interval(s) span more than the {self.block_width}-sample '
                f'block width (largest {max_spacing}), so that much audio falls between blocks. '
                f'Dropped GoPro frames; the audio is missing, not recoverable here.'
            )

        logmel, logmel_ts = self._diagnostic_logmel(piezo, piezo_ts)

        out_grp = ep.require_group('annotations').require_group('contact_audio')
        for name in ('frame_blocks', 'frame_block_start_idx', 'logmel', 'logmel_timestamps'):
            if name in out_grp:
                del out_grp[name]
        out_grp.create_array('frame_blocks', data=blocks, compressor=_BLOSC)
        out_grp.create_array('frame_block_start_idx', data=starts, compressor=_BLOSC)
        out_grp.create_array('logmel', data=logmel, compressor=_BLOSC)
        out_grp.create_array('logmel_timestamps', data=logmel_ts, compressor=_BLOSC)

        covered = int(min(starts[-1] + self.block_width, len(piezo)) - starts[0]) if n_frames else 0
        out_grp.attrs['sample_rate_hz'] = self.sample_rate_hz
        out_grp.attrs['samples_per_gopro_frame'] = self.block_width
        out_grp.attrs['block_alignment'] = self.block_alignment
        out_grp.attrs['nominal_samples_per_frame'] = float(np.median(spacing)) if len(spacing) else 0.0
        out_grp.attrs['max_frame_spacing_samples'] = max_spacing
        out_grp.attrs['n_frames'] = n_frames
        out_grp.attrs['n_frame_gaps'] = n_gaps
        out_grp.attrs['n_zero_filled_samples'] = int(n_zero_filled)
        out_grp.attrs['coverage'] = float(covered) / len(piezo) if len(piezo) else 0.0
        out_grp.attrs['rms'] = float(np.sqrt(np.mean(np.square(piezo, dtype=np.float64))))
        for key, value in self.logmel_cfg.items():
            out_grp.attrs[f'logmel_{key}'] = value

        log.info(
            f'{episode_key}: contact audio — {n_frames} block(s) of {self.block_width} @ '
            f'{self.sample_rate_hz} Hz (spacing {max_spacing} max, {n_gaps} gap(s)), '
            f'rms {out_grp.attrs["rms"]:.4f}, logmel {logmel.shape}'
        )

    def _check_sample_rate(self, episode_key: str, piezo_ts: np.ndarray) -> None:
        """Raise if the timestamps disagree with the configured rate, or are not sorted."""
        diffs = np.diff(piezo_ts)
        # Dropping non-positive deltas instead of rejecting them would let duplicate or
        # backward timestamps through whenever the remaining positive ones still imply the
        # right rate — and `_gather_blocks`' searchsorted assumes piezo_ts is fully sorted, so
        # such input would silently anchor blocks at the wrong samples.
        if len(diffs) == 0 or (diffs <= 0).any():
            raise RuntimeError(
                f'{episode_key}: piezo timestamps are not strictly increasing; cannot infer sample rate.'
            )
        inferred = 1.0 / float(np.median(diffs))
        if abs(inferred - self.sample_rate_hz) > self.sample_rate_hz * _SAMPLE_RATE_TOLERANCE:
            raise RuntimeError(
                f'{episode_key}: piezo timestamps imply {inferred:.1f} Hz but config/contact_audio.yaml '
                f'says {self.sample_rate_hz} Hz. The block width is derived from the configured rate, '
                f'so exporting on this mismatch would give every step the wrong duration of audio.'
            )

    def _gather_blocks(self, piezo: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, int]:
        """Fixed-width block per anchor, zero-filled past the end of the recording."""
        offsets = np.arange(self.block_width, dtype=np.int64)
        idx = starts[:, None] + offsets[None, :]
        in_range = idx < len(piezo)
        blocks = np.where(in_range, piezo[np.clip(idx, 0, len(piezo) - 1)], np.float32(0.0))
        return blocks.astype(np.float32), int((~in_range).sum())

    def _diagnostic_logmel(self, piezo: np.ndarray, piezo_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous log-mel over the whole episode, with hop-centre timestamps."""
        cfg = self.logmel_cfg
        logmel = log_mel_spectrogram(
            piezo,
            self.sample_rate_hz,
            n_fft=int(cfg['n_fft']),
            hop_length=int(cfg['hop_length']),
            win_length=int(cfg['win_length']),
            n_mels=int(cfg['n_mels']),
            fmin=float(cfg['fmin']),
            fmax=float(cfg['fmax']),
            log_offset=float(cfg['log_offset']),
        )
        hop_s = int(cfg['hop_length']) / self.sample_rate_hz
        centre_s = int(cfg['n_fft']) / (2 * self.sample_rate_hz)
        logmel_ts = piezo_ts[0] + centre_s + np.arange(len(logmel), dtype=np.float64) * hop_s
        return logmel, logmel_ts
