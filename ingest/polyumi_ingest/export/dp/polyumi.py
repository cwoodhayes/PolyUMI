"""
``export-polyumi``: the visuomotor ReplayBuffer plus PolyUMI's extra observation streams.

A thin frontend over :func:`export_scenes_to_dp`, not a second exporter. Everything hard —
episode segmentation around pose dropouts, the post-chirp start trim, pose-source resolution,
provenance — is the same code path ``export-dp`` runs; this module only decides which
modalities ride along. A new stream is one more entry in :data:`POLYUMI_MODALITIES`.
"""

from __future__ import annotations

import pathlib

from polyumi_ingest.export.dp.audio import PiezoMicModality
from polyumi_ingest.export.dp.buffer import MIN_SEGMENT_STEPS, export_scenes_to_dp
from polyumi_ingest.export.dp.modality import ExportModality

#: Modalities ``export-polyumi`` adds on top of the visuomotor keys. The contact mic is the
#: first; the finger camera and anything later joins this tuple and nothing else changes.
POLYUMI_MODALITIES: tuple[type[ExportModality], ...] = (PiezoMicModality,)


def export_scenes_to_polyumi(
    scene_paths: list[pathlib.Path],
    output_path: pathlib.Path,
    enforce_preprocessing: bool = True,
    min_segment_steps: int = MIN_SEGMENT_STEPS,
) -> tuple[int, list[dict]]:
    """
    Export one or more pzarr scenes to a ``.zarr.zip`` carrying every PolyUMI modality.

    Identical to :func:`export_scenes_to_dp` in every respect except the extra ``data/`` keys
    (see :data:`POLYUMI_MODALITIES`) and the ``meta.attrs`` describing them. Returns
    ``(n_episodes, provenance)``; each provenance entry gains a ``modalities`` sub-dict.
    """
    return export_scenes_to_dp(
        scene_paths,
        output_path,
        enforce_preprocessing=enforce_preprocessing,
        min_segment_steps=min_segment_steps,
        modalities=[cls() for cls in POLYUMI_MODALITIES],
    )
