"""
Tests for the Phase 3 dataset builder.

``build_dataset`` is a thin wrapper around ``polyumi_ingest.export.dp.export_scenes_to_dp``
(exercised for real in ingest's own test suite) plus the catalog-specific glue: validation,
writing the dataset manifest (§3.2), and upserting Dataset/DatasetMember rows. So here we
monkeypatch the ingest exporter and test the glue this module actually adds.
"""

from __future__ import annotations

import pathlib

import pytest
from polyumi_catalog.dataset_builder import DatasetBuildError, build_dataset
from polyumi_catalog.db import get_engine
from polyumi_catalog.manifests import DatasetManifest
from polyumi_catalog.models import Dataset, DatasetMember, Scene, Task
from sqlmodel import Session as DBSession
from sqlmodel import select


def _seed_scenes(db: DBSession, tmp_path: pathlib.Path) -> tuple[Scene, Scene]:
    a = Scene(scene_id='scene-a', dir=str(tmp_path / 'scene_a'))
    b = Scene(scene_id='scene-b', dir=str(tmp_path / 'scene_b'))
    db.add(a)
    db.add(b)
    db.commit()
    db.refresh(a)
    db.refresh(b)
    return a, b


def test_build_dataset_rejects_empty_name(tmp_path: pathlib.Path):
    """A blank/whitespace name is rejected before touching export or disk."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        with pytest.raises(DatasetBuildError, match='empty'):
            build_dataset(db, name='   ', task_id=None, scene_ids=['scene-a'], output_dir=tmp_path)


def test_build_dataset_rejects_duplicate_name(tmp_path: pathlib.Path):
    """A dataset name that already exists is rejected rather than silently overwritten."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        db.add(Dataset(name='existing'))
        db.commit()
        with pytest.raises(DatasetBuildError, match='already exists'):
            build_dataset(db, name='existing', task_id=None, scene_ids=['scene-a'], output_dir=tmp_path)


def test_build_dataset_rejects_no_scenes(tmp_path: pathlib.Path):
    """Building with zero selected scenes is rejected, not a silent empty dataset."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        with pytest.raises(DatasetBuildError, match='at least one scene'):
            build_dataset(db, name='ds', task_id=None, scene_ids=[], output_dir=tmp_path)


def test_build_dataset_rejects_unknown_scene_id(tmp_path: pathlib.Path):
    """An unknown scene_id is rejected instead of silently skipping it."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        with pytest.raises(DatasetBuildError, match='No such scene'):
            build_dataset(db, name='ds', task_id=None, scene_ids=['no-such-scene'], output_dir=tmp_path)


def test_build_dataset_rejects_unknown_task_id(tmp_path: pathlib.Path):
    """An unknown task_id is rejected rather than silently building an untagged dataset."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        _seed_scenes(db, tmp_path)
        with pytest.raises(DatasetBuildError, match='No such task'):
            build_dataset(db, name='ds', task_id=999, scene_ids=['scene-a'], output_dir=tmp_path)


def test_build_dataset_success_writes_manifest_and_rows(tmp_path: pathlib.Path, monkeypatch):
    """A successful build writes the buffer + manifest and upserts Dataset/DatasetMember rows."""
    calls = []

    def fake_export_scenes_to_dp(scene_paths, output_path):
        calls.append((list(scene_paths), output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b'fake-zip')
        return 5

    monkeypatch.setattr('polyumi_ingest.export.dp.export_scenes_to_dp', fake_export_scenes_to_dp)

    engine = get_engine(tmp_path / 'catalog.db')
    output_dir = tmp_path / 'datasets'
    with DBSession(engine) as db:
        task = Task(name='fold_towel')
        db.add(task)
        db.commit()
        db.refresh(task)
        scene_a, scene_b = _seed_scenes(db, tmp_path)

        dataset = build_dataset(
            db,
            name='fold_towel_v1',
            task_id=task.id,
            scene_ids=[scene_a.scene_id, scene_b.scene_id],
            output_dir=output_dir,
        )

        assert dataset.n_episodes == 5
        assert dataset.task_id == task.id
        expected_paths = [pathlib.Path(scene_a.dir), pathlib.Path(scene_b.dir)]
        assert calls == [(expected_paths, output_dir / 'fold_towel_v1.zarr.zip')]

        members = db.exec(select(DatasetMember).where(DatasetMember.dataset_id == dataset.id)).all()
        assert {m.scene_id for m in members} == {'scene-a', 'scene-b'}
        assert all(m.episodes == 'all' for m in members)

    manifest_path = output_dir / 'fold_towel_v1.dataset.json'
    assert manifest_path.is_file()
    manifest = DatasetManifest.from_file(manifest_path)
    assert manifest.n_episodes == 5
    assert manifest.task == 'fold_towel'
    assert {m.scene_id for m in manifest.members} == {'scene-a', 'scene-b'}
    assert (output_dir / 'fold_towel_v1.zarr.zip').is_file()


def test_build_dataset_export_failure_leaves_no_manifest_or_rows(tmp_path: pathlib.Path, monkeypatch):
    """If the underlying export raises, no manifest/buffer/DB rows are left behind."""

    def failing_export(scene_paths, output_path):
        raise RuntimeError('no EPISODE sessions to export across the given scene(s).')

    monkeypatch.setattr('polyumi_ingest.export.dp.export_scenes_to_dp', failing_export)

    engine = get_engine(tmp_path / 'catalog.db')
    output_dir = tmp_path / 'datasets'
    with DBSession(engine) as db:
        scene_a, _ = _seed_scenes(db, tmp_path)
        with pytest.raises(DatasetBuildError, match='no EPISODE sessions'):
            build_dataset(db, name='broken', task_id=None, scene_ids=[scene_a.scene_id], output_dir=output_dir)

        assert db.exec(select(Dataset).where(Dataset.name == 'broken')).first() is None

    assert not (output_dir / 'broken.dataset.json').exists()
    assert not (output_dir / 'broken.zarr.zip').exists()
