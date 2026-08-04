"""Tests for engine setup and the startup schema check over an older catalog DB."""

from __future__ import annotations

import pathlib

import pytest
import typer
from polyumi_catalog.db import get_engine, rebuild_schema, schema_mismatches
from polyumi_catalog.main import _open_db
from polyumi_catalog.models import Scene, Session
from sqlalchemy import text
from sqlmodel import Session as DBSession
from sqlmodel import select


def _seeded_engine(tmp_path: pathlib.Path):
    """Build an engine with one scene + session row, so a rebuild is observable."""
    engine = get_engine(tmp_path / 'catalog.db')
    with DBSession(engine) as db:
        db.add(Scene(scene_id='scene-1', dir=str(tmp_path / 'scene_1')))
        db.add(Session(session_id='sess-1', scene_id='scene-1', dir='session_1', session_type='EPISODE'))
        db.commit()
    return engine


def _drop_a_column(engine) -> None:
    """Age the DB back to a schema that predates a column, as an older checkout would leave it."""
    with engine.begin() as conn:
        conn.execute(text('ALTER TABLE session DROP COLUMN slam_attrs_json'))


def test_schema_mismatches_is_empty_for_a_current_db(tmp_path: pathlib.Path):
    """A DB just created from the models reports no drift."""
    assert schema_mismatches(get_engine(tmp_path / 'catalog.db')) == []


def test_schema_mismatches_reports_missing_and_leftover_columns(tmp_path: pathlib.Path):
    """Both directions of drift are named, so the prompt can say what's actually wrong."""
    engine = _seeded_engine(tmp_path)
    _drop_a_column(engine)
    with engine.begin() as conn:
        conn.execute(text('ALTER TABLE scene ADD COLUMN retired_field TEXT'))

    assert schema_mismatches(engine) == [
        'scene.retired_field is left over from an older schema',
        'session.slam_attrs_json is missing',
    ]


def test_rebuild_schema_empties_the_db_and_clears_the_drift(tmp_path: pathlib.Path):
    """Rebuilding drops the cached rows — they all come back from disk on the next sync."""
    engine = _seeded_engine(tmp_path)
    _drop_a_column(engine)

    rebuild_schema(engine)

    assert schema_mismatches(engine) == []
    with DBSession(engine) as db:
        assert db.exec(select(Session)).all() == []


def test_open_db_rebuilds_a_stale_db_when_confirmed(tmp_path: pathlib.Path, monkeypatch):
    """The CLI offers the rebuild, and on yes hands back a clean engine flagged for re-sync."""
    engine = _seeded_engine(tmp_path)
    _drop_a_column(engine)
    monkeypatch.setattr(typer, 'confirm', lambda *a, **k: True)

    engine, was_rebuilt = _open_db(tmp_path / 'catalog.db', rebuild_if_stale=None)

    assert was_rebuilt is True
    assert schema_mismatches(engine) == []


def test_open_db_exits_when_the_rebuild_is_declined(tmp_path: pathlib.Path, monkeypatch):
    """Declining stops rather than running queries against a DB that would fail mid-request."""
    engine = _seeded_engine(tmp_path)
    _drop_a_column(engine)
    monkeypatch.setattr(typer, 'confirm', lambda *a, **k: False)

    with pytest.raises(typer.Exit):
        _open_db(tmp_path / 'catalog.db', rebuild_if_stale=None)


def test_open_db_leaves_a_current_db_alone(tmp_path: pathlib.Path, monkeypatch):
    """No drift means no prompt and no data loss — the common case must not touch the rows."""
    _seeded_engine(tmp_path)
    monkeypatch.setattr(typer, 'confirm', lambda *a, **k: pytest.fail('should not have prompted'))

    engine, was_rebuilt = _open_db(tmp_path / 'catalog.db', rebuild_if_stale=None)

    assert was_rebuilt is False
    with DBSession(engine) as db:
        assert len(db.exec(select(Session)).all()) == 1
