"""
Multi-scene dataset builder & export (Phase 3).

Combines whole scenes into one UMI-format ReplayBuffer, per the "ingest owns
preprocessing/export, catalog only imports it" decision (docs/catalog-ui-plan.md §10.2) — the
same pattern already used for Phase 2.5's MCAP export. Exports the buffer first and only then
writes the dataset manifest (§3.2) beside it, so a failed/interrupted export never leaves a
manifest pointing at nonexistent data. The manifest is the authoritative record; the DB rows
are a cache of it (rebuildable by ``sync.sync_datasets``, same principle as scene->task in §3.1).

Dataset membership is whole-scene only for now (§10 decision 1) — every member's ``episodes``
is ``"all"``; per-episode selection is a non-breaking extension the schema already allows.
"""

from __future__ import annotations

import pathlib
import subprocess

from sqlmodel import Session as DBSession
from sqlmodel import select

from polyumi_catalog.manifests import DatasetManifest, DatasetMemberSpec
from polyumi_catalog.models import Dataset, DatasetMember, Scene, Task


class DatasetBuildError(ValueError):
    """A dataset build was rejected (invalid input) or the underlying export failed."""


def _repo_git_hash() -> str | None:
    """Best-effort short-circuit git hash of this checkout, for the manifest's provenance field."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_dataset(
    db: DBSession,
    *,
    name: str,
    task_id: int | None,
    scene_ids: list[str],
    output_dir: pathlib.Path,
) -> Dataset:
    """
    Export ``scene_ids`` into one UMI ReplayBuffer named ``name`` and record it as a Dataset.

    Writes ``<output_dir>/<name>.zarr.zip`` and its provenance manifest
    ``<output_dir>/<name>.dataset.json`` before touching the DB, then upserts the Dataset +
    DatasetMember rows. Raises :class:`DatasetBuildError` for invalid input or if the export
    itself fails (in which case nothing is written and no DB rows are created).
    """
    from polyumi_ingest.export.dp import export_scenes_to_dp

    name = name.strip()
    if not name:
        raise DatasetBuildError('Dataset name cannot be empty.')
    if db.exec(select(Dataset).where(Dataset.name == name)).first() is not None:
        raise DatasetBuildError(f'Dataset {name!r} already exists.')
    if not scene_ids:
        raise DatasetBuildError('Select at least one scene.')

    scenes: list[Scene] = []
    for scene_id in scene_ids:
        scene = db.get(Scene, scene_id)
        if scene is None:
            raise DatasetBuildError(f'No such scene: {scene_id}')
        scenes.append(scene)

    task_name = None
    if task_id is not None:
        task = db.get(Task, task_id)
        if task is None:
            raise DatasetBuildError(f'No such task: {task_id}')
        task_name = task.name

    output_path = output_dir / f'{name}.zarr.zip'
    manifest_path = output_dir / f'{name}.dataset.json'

    try:
        n_episodes, pose_provenance = export_scenes_to_dp([pathlib.Path(s.dir) for s in scenes], output_path)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        raise DatasetBuildError(str(err)) from err

    manifest = DatasetManifest(
        name=name,
        task=task_name,
        output=output_path.name,
        n_episodes=n_episodes,
        polyumi_version=_repo_git_hash(),
        members=[DatasetMemberSpec(scene_id=s.scene_id, scene_dir=s.dir, episodes='all') for s in scenes],
        pose_provenance=pose_provenance,
    )
    manifest.to_file(manifest_path)

    dataset = Dataset(
        name=name,
        task_id=task_id,
        manifest_path=str(manifest_path),
        output_path=str(output_path),
        n_episodes=n_episodes,
        polyumi_version=manifest.polyumi_version,
    )
    db.add(dataset)
    db.flush()  # assign dataset.id
    for scene in scenes:
        db.add(DatasetMember(dataset_id=dataset.id, scene_id=scene.scene_id, episodes='all'))
    db.commit()
    db.refresh(dataset)
    return dataset
