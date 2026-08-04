"""
Engine / session helpers for the catalog SQLite cache.

There are deliberately **no migrations** here. Every row is derived from the recordings
tree, so when the models and an existing DB file disagree the answer is always to throw
the file away and re-sync — :func:`schema_mismatches` detects the drift at startup and
the CLI offers the rebuild. Adding a column is therefore just editing ``models.py``.
"""

from __future__ import annotations

import logging
import pathlib

from sqlalchemy import Engine, inspect
from sqlmodel import SQLModel, create_engine

# importing models registers the tables on SQLModel.metadata
from polyumi_catalog import models  # noqa: F401

log = logging.getLogger('catalog.db')


def default_db_path(recordings_dir: pathlib.Path) -> pathlib.Path:
    """Return the default catalog DB path for a recordings tree (``<rec>/.catalog/catalog.db``)."""
    return recordings_dir / '.catalog' / 'catalog.db'


def default_datasets_dir(recordings_dir: pathlib.Path) -> pathlib.Path:
    """Return the default directory exported datasets + their manifests live in."""
    return recordings_dir / 'datasets'


def schema_mismatches(engine: Engine) -> list[str]:
    """
    Describe every way the DB's schema differs from the models; empty means it's current.

    ``create_all`` creates missing *tables* but never touches an existing one, so a DB
    written before a column was added keeps opening fine and then fails with a bare
    ``no such column`` on the first query that mentions it. Callers check this at startup
    and rebuild rather than patching the file — see the module docstring.

    Column *types* aren't compared: SQLite's are advisory anyway, and the presence set is
    what actually breaks queries.
    """
    inspector = inspect(engine)
    db_tables = set(inspector.get_table_names())
    model_tables = {t.name for t in SQLModel.metadata.sorted_tables}
    out = [f'table {name} is missing' for name in sorted(model_tables - db_tables)]
    out += [f'table {name} is left over from an older schema' for name in sorted(db_tables - model_tables)]
    for table in SQLModel.metadata.sorted_tables:
        if table.name not in db_tables:
            continue
        db_columns = {c['name'] for c in inspector.get_columns(table.name)}
        model_columns = {c.name for c in table.columns}
        out += [f'{table.name}.{c} is missing' for c in sorted(model_columns - db_columns)]
        out += [f'{table.name}.{c} is left over from an older schema' for c in sorted(db_columns - model_columns)]
    return out


def rebuild_schema(engine: Engine) -> None:
    """
    Drop every table and recreate it empty, discarding the cache.

    Safe by construction: the recordings tree and the dataset manifests are the source of
    truth for everything in here, so a sync repopulates it. Callers are expected to run one
    immediately afterwards.
    """
    log.warning('Rebuilding the catalog DB from scratch; re-syncing from disk.')
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


def get_engine(db_path: pathlib.Path) -> Engine:
    """
    Create (or open) the SQLite engine at ``db_path`` and ensure the schema exists.

    The parent directory is created if missing. This creates missing tables but does not
    reconcile an existing one against the models — call :func:`schema_mismatches` for that.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f'sqlite:///{db_path}')
    SQLModel.metadata.create_all(engine)
    return engine
