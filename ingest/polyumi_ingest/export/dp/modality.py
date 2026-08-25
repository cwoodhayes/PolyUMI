"""
The seam by which an exporter contributes extra ``data/<key>`` arrays to a ReplayBuffer.

``pingest export`` (``--type dp``, the default) emits the visuomotor keys and nothing else.
``--type polyumi`` emits those plus whatever modalities it is configured with — audio today, the
finger camera later. Both run the *same* ``_export_episode``: the segmentation, chirp trim, pose
resolution and provenance are hard enough to get right once, and a second copy would drift.

A modality is one instance per export run, so ``prepare_episode`` may stash on ``self`` whatever
``segment_arrays`` needs — the same shape as ``PreprocessingStep.prepare_scene``. An export with
no modalities is byte-identical to one from before this seam existed, down to the provenance
sidecar, which is what lets ``--type dp`` keep its contract unchanged.
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
