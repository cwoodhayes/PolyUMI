"""
Full preprocessing pipeline status + trigger for the scene detail pane (Phase 4).

``scene_pp_status`` mirrors what ``pingest pp --list`` plus a scene's
``preprocessing_steps`` attr would tell you: which registered steps exist and
which of them are already marked complete on this scene's pzarr. ``run_full_pipeline``
mirrors `pingest pp` called with no step argument — ``ensure_pzarr`` first (build the
store, or extend it with sessions recorded since it was built, requiring every session's
gopro.mp4 sidecar exactly as the CLI does without --skip-gopro), then run every step in
order, skipping ones already complete. No new pipeline logic lives here; this only reuses
ingest's own ``ensure_pzarr`` / ``run_preprocessing`` / ``available_preprocessing_steps``,
per the "ingest owns preprocessing/export, catalog only imports it" split.
"""

from __future__ import annotations

import logging
import pathlib

log = logging.getLogger('catalog.pp_status')


def scene_pp_status(scene_dir: pathlib.Path) -> dict:
    """
    Return the pzarr-build + per-step completion status for scene_dir.

    Reads the ``preprocessing_steps`` attr directly off the root group rather than going
    through ``inspect_pzarr`` (which reads every episode's full per-sample timestamp
    arrays to compute stream shapes/rates this caller doesn't need) — this is called on
    every scene selection, so it needs to stay a cheap, attrs-only zarr open.
    """
    import zarr

    from polyumi_ingest.preproc import available_preprocessing_steps, preprocessing_step_versions
    from polyumi_ingest.pzarr.scene_files import SceneFiles

    all_steps = available_preprocessing_steps()
    zarr_path = SceneFiles.resolve_zarr_path(scene_dir)
    if not zarr_path.exists():
        return {
            'pzarr_exists': False,
            'steps': [
                {'number': s.step_number, 'name': s.step_name, 'complete': False, 'git_sha': None, 'completed_at': None}
                for s in all_steps
            ],
            'n_complete': 0,
            'n_total': len(all_steps),
        }

    root = zarr.open_group(str(zarr_path), mode='r')
    completed = {int(n) for n in root.attrs.get('preprocessing_steps', [])}
    versions = preprocessing_step_versions(root)
    steps = []
    for s in all_steps:
        # None (not 'unknown') for steps processed before per-step provenance existed, so the
        # template can render "—" rather than implying the commit was looked up and missed.
        version = versions.get(str(s.step_number)) or {}
        steps.append(
            {
                'number': s.step_number,
                'name': s.step_name,
                'complete': s.step_number in completed,
                'git_sha': version.get('git_sha'),
                'completed_at': version.get('completed_at'),
            }
        )
    return {
        'pzarr_exists': True,
        'steps': steps,
        # counted from `steps` (intersected with currently-registered step numbers), not
        # len(completed) directly — a scene processed under a since-retired step number
        # would otherwise report e.g. "7/5 complete".
        'n_complete': sum(1 for s in steps if s['complete']),
        'n_total': len(all_steps),
    }


def reset_pp_status(scene_dir: pathlib.Path) -> None:
    """
    Clear scene_dir's recorded step completion so every step reads as incomplete again.

    Called before a forced re-run. Two reasons, one cosmetic and one not:

    * The pane would otherwise sit at "5 / 5 complete" for the entire run, giving no
      indication anything is happening beyond the "Running pipeline…" line — steps only
      re-tick as each finishes, and they were already ticked.
    * If a forced run dies at step 2, the pzarr would still advertise steps 3-5 as
      complete even though their outputs were computed from the *previous* step 2. Marks
      cleared up front make that state honest: what's re-ticked is what's actually been
      re-run against the current inputs.

    Only the completion marks are cleared. The step outputs themselves stay in place until
    each step overwrites them, and the pzarr is not rebuilt — rebuilding means re-encoding
    every frame, which a preprocessing re-run has no reason to pay for.

    A no-op if the pzarr doesn't exist yet (nothing to reset).

    The clearing itself is ingest's ``clear_step_marks``, not a second implementation here:
    it has to reach the per-episode marks as well as the root's, and those gate ingest's own
    episode loop. This function only exists because the route wants the marks gone *before*
    it renders its response, without waiting for the pipeline thread.
    """
    import zarr

    from polyumi_ingest.preproc import clear_step_marks
    from polyumi_ingest.pzarr.scene_files import SceneFiles

    zarr_path = SceneFiles.resolve_zarr_path(scene_dir)
    if not zarr_path.exists():
        return
    clear_step_marks(zarr.open_group(str(zarr_path), mode='a'))


def run_full_pipeline(scene_dir: pathlib.Path, force: bool = False) -> None:
    """
    Run the complete preprocessing pipeline on scene_dir, building pzarr first if needed.

    ``force`` is passed straight through to ``run_preprocessing``: without it, a step
    already marked complete is skipped (the "continue" button — safe to click even on a
    fully-processed scene, since it's then a no-op); with it, every step re-runs from
    scratch regardless of completion, discarding whatever it previously wrote (the
    "re-run" button — e.g. to pick up a preprocessing code change on an already-processed
    scene). A forced run clears the recorded completion marks first — inside
    ``run_preprocessing`` now, so the CLI's ``--force`` gets it too — and progress reads
    from zero as the run proceeds.

    Blocks for as long as the pipeline takes — SLAM in particular can take minutes —
    so callers should run this on a background thread rather than the request thread.
    Raises FileNotFoundError/RuntimeError/NotImplementedError/KeyError on failure, same
    as ingest's own build_pzarr/run_preprocessing.
    """
    from polyumi_ingest.preproc import run_preprocessing
    from polyumi_ingest.pzarr import ensure_pzarr

    # Same call `pingest pp` makes, so the button and the terminal agree on what "build first"
    # means — including extending a store whose scene has grown since it was built. Doing that
    # decision separately here is what let the Fetch button pull new sessions that Run pp then
    # never put in the store.
    ensure_pzarr(scene_dir)
    # No reset_pp_status here: run_preprocessing clears the marks for the steps it is about
    # to force, root and per-episode both. The route calls reset_pp_status itself beforehand,
    # for its own rendering reasons.
    run_preprocessing(scene_dir, step_number=None, force=force)
