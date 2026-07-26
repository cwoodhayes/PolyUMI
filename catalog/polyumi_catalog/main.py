"""
polyumi-catalog CLI: manage the metadata catalog over the recordings tree.

Phase 0 exposes ``sync`` (scan recordings → SQLite cache). The web UI (``serve``)
arrives in Phase 1.
"""

from __future__ import annotations

import logging
import os
import pathlib

import typer
from polyumi_pi.files.session import DEFAULT_RECORDINGS_DIR
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from polyumi_catalog.db import default_db_path, get_engine
from polyumi_catalog.sync import sync_recordings

logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
    format='%(message)s',
    handlers=[RichHandler(show_time=True, show_level=True, show_path=False, rich_tracebacks=True)],
)
log = logging.getLogger('catalog')

app = typer.Typer(help='PolyUMI metadata catalog & dataset builder.')


@app.callback()
def _main():
    """PolyUMI metadata catalog & dataset builder."""
    # Present so Typer keeps sub-command dispatch (``serve`` arrives in Phase 1).


@app.command()
def sync(
    recordings: pathlib.Path = typer.Option(
        DEFAULT_RECORDINGS_DIR,
        '--recordings',
        '-r',
        help='Recordings directory to scan.',
    ),
    db: pathlib.Path | None = typer.Option(
        None,
        '--db',
        help='Catalog SQLite path. Defaults to <recordings>/.catalog/catalog.db.',
    ),
    force: bool = typer.Option(
        False,
        '--force',
        help='Re-parse every scene, ignoring mtime gating.',
    ),
):
    """Scan the recordings tree and update the catalog cache."""
    recordings = recordings.expanduser()
    if not recordings.is_dir():
        log.error(f'Recordings directory not found: {recordings}')
        raise typer.Exit(1)

    db_path = db.expanduser() if db else default_db_path(recordings)
    engine = get_engine(db_path)
    stats = sync_recordings(recordings, engine, force=force)

    console = Console()
    table = Table(title='Catalog sync', title_justify='left', header_style='bold')
    table.add_column('Metric')
    table.add_column('Count', justify='right')
    table.add_row('scenes scanned', str(stats.scenes_scanned))
    table.add_row('scenes updated', str(stats.scenes_updated))
    table.add_row('scenes skipped', str(stats.scenes_skipped))
    table.add_row('sessions upserted', str(stats.sessions_upserted))
    table.add_row('sessions removed', str(stats.sessions_removed))
    table.add_row('tasks created', str(stats.tasks_created))
    table.add_row('task conflicts', str(len(stats.conflicts)))
    console.print()
    console.print(f'DB: [bold]{db_path}[/bold]')
    console.print(table)

    if stats.conflicts:
        console.print('\n[yellow]Task conflicts (metadata.json vs scene.json):[/yellow]')
        for c in stats.conflicts:
            console.print(f'  {c.session_id[:8]}  scene={c.scene_task!r}  meta={c.meta_task!r}')


if __name__ == '__main__':
    app()
