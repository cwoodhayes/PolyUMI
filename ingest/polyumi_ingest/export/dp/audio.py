"""The piezo contact-mic modality: preprocessing step 6's blocks become ``data/mic_0``."""

from __future__ import annotations

import numpy as np
import zarr

from polyumi_ingest.config import load_contact_audio_config
from polyumi_ingest.export.dp.modality import ExportModality
from polyumi_ingest.preproc.contact_audio_step import ContactAudioStep
from polyumi_ingest.pzarr.store import arr, grp

#: UMI-style key for the contact mic, matching ManiWAV's naming so their dataset and encoder
#: read our buffers unmodified. Their ``mic_1`` (the air mic) is deliberately not exported —
#: it exists to carry the sync chirp, not to be an observation.
MIC_KEY = 'mic_0'

_BLOCKS_PATH = 'annotations/contact_audio/frame_blocks'


class PiezoMicModality(ExportModality):
    """
    Contact-mic waveform on the exported step grid, as ``mic_0`` of shape ``(T, stride * B)``.

    Raw waveform, not a spectrogram, on purpose: the policy's log-mel is computed in the training
    container after augmentation that only exists in the waveform domain, and the mel parameters
    are hyperparameters we would otherwise be freezing at ingest time. See
    ``docs/maniwav-audio-policy.md``.

    Each row is the ``stride`` per-frame blocks belonging to one step, concatenated. Step 6
    anchors every frame's block at that frame's own timestamp with a width no smaller than the
    largest gap between anchors, so consecutive blocks abut or overlap — which is what makes the
    concatenation a gapless waveform, and in turn what ManiWAV's ``down_sample_steps: 1`` audio
    path assumes when it flattens the rows back into one signal before the mel.

    Whether a step's audio comes from before or after its observation instant is
    ``block_alignment`` in ``config/contact_audio.yaml``. That choice is a row shift applied here
    rather than in step 6, so changing it is a re-export rather than a re-preprocess.
    """

    name = MIC_KEY
    required_steps = frozenset({ContactAudioStep.step_number})

    def __init__(self) -> None:
        """Read the block geometry once, so every episode in a buffer is cut the same way."""
        blocks = load_contact_audio_config()['blocks']
        self.sample_rate_hz = int(blocks['sample_rate_hz'])
        self.block_width = int(blocks['samples_per_gopro_frame'])
        self.alignment = str(blocks['block_alignment'])
        self._blocks: np.ndarray | None = None
        self._stride = 1

    def prepare_episode(self, ep: zarr.Group, episode_key: str, gopro_ts: np.ndarray, stride: int) -> None:
        """Load this session's per-frame blocks and check they match the configured geometry."""
        if _BLOCKS_PATH not in ep:
            raise RuntimeError(
                f'{episode_key}: no {_BLOCKS_PATH} — run `pingest pp 6` (contact-audio) before '
                f'exporting with export-polyumi.'
            )
        blocks_arr = arr(ep, _BLOCKS_PATH)
        n, width = blocks_arr.shape
        if n != len(gopro_ts):
            raise RuntimeError(
                f'{episode_key}: {_BLOCKS_PATH} has {n} row(s) but the GoPro grid has '
                f'{len(gopro_ts)}. Re-run `pingest pp 6 --force`.'
            )
        if width != self.block_width:
            raise RuntimeError(
                f'{episode_key}: {_BLOCKS_PATH} is {width} samples wide but '
                f'config/contact_audio.yaml says {self.block_width}. The width is the mic_0 '
                f'contract, so re-run `pingest pp 6 --force` rather than exporting a mix.'
            )
        stored_rate = int(grp(ep, 'annotations/contact_audio').attrs.get('sample_rate_hz', self.sample_rate_hz))
        if stored_rate != self.sample_rate_hz:
            raise RuntimeError(
                f'{episode_key}: blocks were written at {stored_rate} Hz but config says {self.sample_rate_hz} Hz.'
            )
        self._blocks = np.asarray(blocks_arr[:], dtype=np.float32)
        self._stride = stride

    def segment_arrays(self, gidx: np.ndarray) -> dict[str, np.ndarray]:
        """Concatenate each step's ``stride`` frame-blocks into one ``mic_0`` row."""
        assert self._blocks is not None, 'prepare_episode must run before segment_arrays'
        stride = self._stride
        if self.alignment == 'causal':
            # The stride frames ENDING at the step: only audio that exists by the time the policy
            # is asked to act.
            rows = gidx[:, None] - stride + 1 + np.arange(stride)[None, :]
        else:
            # ManiWAV's convention: the stride frames STARTING at the step, i.e. look-ahead.
            rows = gidx[:, None] + np.arange(stride)[None, :]

        # Rows falling outside the episode — the first step's history under causal alignment,
        # the last step's look-ahead under forward — are silence, not a repeat of the nearest
        # block. Repeating would splice a copy of real audio into the waveform and read as a
        # genuine contact event; ManiWAV's sampler zero-pads audio at episode start for the same
        # reason, where it edge-repeats RGB.
        in_range = (rows >= 0) & (rows < self._blocks.shape[0])
        gathered = self._blocks[np.clip(rows, 0, self._blocks.shape[0] - 1)]
        gathered[~in_range] = 0.0
        mic = gathered.reshape(len(gidx), stride * self.block_width)
        return {MIC_KEY: mic.astype(np.float32)}

    def segment_provenance(self, gidx: np.ndarray) -> dict:
        """Record the geometry this segment's rows were cut with."""
        return {
            'samples_per_step': self._stride * self.block_width,
            'block_alignment': self.alignment,
        }

    def meta_attrs(self) -> dict:
        """Make the buffer self-describing about the audio contract it was built under."""
        return {
            'mic_0_sample_rate_hz': self.sample_rate_hz,
            'mic_0_samples_per_gopro_frame': self.block_width,
            'mic_0_samples_per_step': self._stride * self.block_width,
            'mic_0_block_alignment': self.alignment,
            'mic_0_source': 'finger/finger_piezo',
        }
