"""
Full preprocessing pipeline status + trigger for the scene detail pane (Phase 4).

``scene_pp_status`` mirrors what ``pingest pp --list`` plus a scene's
``preprocessing_steps`` attr would tell you: which registered steps exist and
which of them are already marked complete on this scene's pzarr. ``run_full_pipeline``
mirrors `pingest pp` called with no step argument — build pzarr first if it doesn't
exist yet (requiring every session's gopro.mp4 sidecar, same as the CLI without
--skip-gopro), then run every step in order, skipping ones already complete. No new
pipeline logic lives here; this only reuses ingest's own ``build_pzarr`` /
``run_preprocessing`` / ``available_preprocessing_steps``, per the "ingest owns
preprocessing/export, catalog only imports it" split (docs/catalog-ui-plan.md §10.2).
"""

from __future__ import annotations

import logging
import pathlib

log = logging.getLogger('catalog.pp_status')


def scene_pp_status(scene_dir: pathlib.Path) -> dict:
    """Return the pzarr-build + per-step completion status for scene_dir."""
    from polyumi_ingest.preproc import available_preprocessing_steps
    from polyumi_ingest.pzarr import inspect_pzarr
    from polyumi_ingest.pzarr.scene_files import SceneFiles

    all_steps = available_preprocessing_steps()
    zarr_path = SceneFiles.resolve_zarr_path(scene_dir)
    if not zarr_path.exists():
        return {
            'pzarr_exists': False,
            'steps': [{'number': s.step_number, 'name': s.step_name, 'complete': False} for s in all_steps],
            'n_complete': 0,
            'n_total': len(all_steps),
        }

    info = inspect_pzarr(scene_dir)
    completed = {int(n) for n in info.attrs.get('preprocessing_steps', [])}
    steps = [{'number': s.step_number, 'name': s.step_name, 'complete': s.step_number in completed} for s in all_steps]
    return {
        'pzarr_exists': True,
        'steps': steps,
        'n_complete': len(completed),
        'n_total': len(all_steps),
    }


def missing_gopro_mp4s(scene_dir: pathlib.Path) -> list[str]:
    """Return session directory names under scene_dir that are missing their gopro.mp4 sidecar."""
    from polyumi_ingest.pzarr import GOPRO_MP4
    from polyumi_ingest.pzarr.scene_files import SceneFiles

    scene = SceneFiles.from_path(scene_dir)
    return [s.path.name for s in scene.sessions if not (s.path / GOPRO_MP4).exists()]


def run_full_pipeline(scene_dir: pathlib.Path) -> None:
    """
    Run the complete preprocessing pipeline on scene_dir, building pzarr first if needed.

    Blocks for as long as the pipeline takes — SLAM in particular can take minutes —
    so callers should run this on a background thread rather than the request thread.
    Raises FileNotFoundError/RuntimeError/NotImplementedError/KeyError on failure, same
    as ingest's own build_pzarr/run_preprocessing.
    """
    from polyumi_ingest.preproc import run_preprocessing
    from polyumi_ingest.pzarr import build_pzarr
    from polyumi_ingest.pzarr.scene_files import SceneFiles

    zarr_path = SceneFiles.resolve_zarr_path(scene_dir)
    if not zarr_path.exists():
        missing = missing_gopro_mp4s(scene_dir)
        if missing:
            raise FileNotFoundError(
                f'Cannot build pzarr for {scene_dir.name}: missing gopro.mp4 in {len(missing)} session(s): '
                + ', '.join(missing)
            )
        log.info(f'No scene.zarr found for {scene_dir.name}; building pzarr first...')
        build_pzarr(scene_dir)

    run_preprocessing(scene_dir, step_number=None)
