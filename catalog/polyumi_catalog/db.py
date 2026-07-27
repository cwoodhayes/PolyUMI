"""Engine / session helpers for the catalog SQLite cache."""

from __future__ import annotations

import pathlib

from sqlalchemy import Engine
from sqlmodel import SQLModel, create_engine

# importing models registers the tables on SQLModel.metadata
from polyumi_catalog import models  # noqa: F401


def default_db_path(recordings_dir: pathlib.Path) -> pathlib.Path:
    """Return the default catalog DB path for a recordings tree (``<rec>/.catalog/catalog.db``)."""
    return recordings_dir / '.catalog' / 'catalog.db'


def default_datasets_dir(recordings_dir: pathlib.Path) -> pathlib.Path:
    """Return the default directory exported datasets + their manifests live in."""
    return recordings_dir / 'datasets'


def get_engine(db_path: pathlib.Path) -> Engine:
    """
    Create (or open) the SQLite engine at ``db_path`` and ensure the schema exists.

    The parent directory is created if missing.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f'sqlite:///{db_path}')
    SQLModel.metadata.create_all(engine)
    return engine
