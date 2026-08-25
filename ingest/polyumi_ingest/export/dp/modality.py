"""
``ExportModality``: the interface a ``--type polyumi`` stream implements to ride along.

Ride along in ``export.dp.buffer``'s exported ReplayBuffer, specifically — see that module's
docstring for how the seam fits into the exporter as a whole (why it's the same code path as the
default ``--type dp``, and what attaching zero modalities guarantees); this module is just the
contract itself.

A modality is one instance per export run, so ``prepare_episode`` may stash on ``self`` whatever
``segment_arrays`` needs — the same shape as ``PreprocessingStep.prepare_scene``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import zarr


class ExportModality(ABC):
    """One extra family of ``data/<key>`` arrays contributed to an exported ReplayBuffer."""

    #: Short label used in error messages and per-segment provenance.
    name: str

    #: Preprocessing steps this modality's inputs come from. Unioned into the set
    #: ``_check_preprocessing_complete`` enforces, so a modality demands exactly the steps it
    #: needs without those steps blocking the visuomotor export.
    required_steps: frozenset[int] = frozenset()

    def prepare_episode(self, ep: zarr.Group, episode_key: str, gopro_ts: np.ndarray, stride: int) -> None:
        """
        Load what one session needs, before its segments are cut.

        Raising here fails the export with a named cause rather than emitting a buffer that is
        quietly missing this modality.
        """

    def valid_steps(self, steps: np.ndarray) -> np.ndarray | None:
        """
        Which of ``steps`` this modality actually has an observation for; ``None`` means all.

        Folded into the same validity mask as missing poses, so a stretch this modality cannot
        cover is *trimmed or split out* by ``_valid_segments`` rather than failing the episode.
        That is the right shape for a sensor that simply stops before the others do — the finger
        camera reliably stops recording ~0.65 s before the GoPro, so rejecting the episode would
        reject every episode, while exporting the span anyway would pair a frozen frame with
        moving proprioception.

        ``steps`` indexes the GoPro frame grid, ascending and stride-spaced, already trimmed to
        the post-chirp span. Return a boolean mask of the same length.
        """
        return None

    @abstractmethod
    def segment_arrays(self, gidx: np.ndarray) -> dict[str, np.ndarray]:
        """
        Build the ``data/<key>`` arrays for one exported segment, each with ``len(gidx)`` rows.

        ``gidx`` indexes the GoPro frame grid, ascending and stride-spaced.
        """

    def segment_provenance(self, gidx: np.ndarray) -> dict:
        """Per-segment provenance fragment, recorded under ``provenance['modalities'][name]``."""
        return {}

    def meta_attrs(self) -> dict:
        """Buffer-wide ``meta.attrs`` entries, merged once after every scene is appended."""
        return {}
