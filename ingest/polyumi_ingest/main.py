"""
ingest/main.py - PolyUMI ingest scripts to deal with pi's file & build pzarr stores.

See docs/data-format.md for an overview of the pzarr format.
"""

import logging
import os
import pathlib
import shutil

import typer
from polyumi_pi.files.session import SessionFiles
from rich.logging import RichHandler
from rich.prompt import Confirm

from polyumi_ingest.gopro_fetch import DEFAULT_THRESHOLD_MS, find_gopro_video
from polyumi_ingest.pi_fetch import PiFetch
from polyumi_ingest.pzarr import FINGER_MP4, GOPRO_MP4
from polyumi_ingest.video_helpers import encode_session_video

logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
    format='%(message)s',
    handlers=[
        RichHandler(
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
        )
    ],
)
log = logging.getLogger('ingest')

app = typer.Typer()


def _human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    unit = 'B'
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024 or unit == 'TB':
            break
        size /= 1024
    return f'{size:.1f} {unit}'


DEFAULT_HOST = 'pi@polyumi-pi.local'

# put this in the root of the repo
DEFAULT_RECORDINGS_DIR = pathlib.Path(__file__).parent.parent.parent / 'recordings'


@app.command()
def fetch(
    host: str = typer.Option(DEFAULT_HOST, help='SSH hostname of the Pi.'),
    output_dir: pathlib.Path = typer.Option(
        DEFAULT_RECORDINGS_DIR,
        help='Local directory to write scenes into.',
    ),
    latest: bool = typer.Option(
        False,
        '--latest',
        help='Only fetch the latest scene.',
    ),
    verbose_transfer: bool = typer.Option(
        False,
        '--verbose-transfer',
        help='Show detailed transfer output for debugging.',
    ),
):
    """Fetch recorded sessions from the Pi via tar-over-ssh."""
    output_dir = output_dir.resolve()
    pi = PiFetch(host)

    if latest:
        scene_name = pi.resolve_latest_scene()
        scenes_to_fetch = [scene_name]
        log.info(f'Latest scene: {scene_name}')
    else:
        log.info(f'Listing scenes on {host}...')
        scenes_to_fetch = pi.list_remote_scenes()
        log.info(f'Found {len(scenes_to_fetch)} scene(s) on {host}.')

    if not scenes_to_fetch:
        log.info('No scenes to fetch.')
        raise typer.Exit()

    # filter out already-fetched scenes
    to_fetch = []
    skipped = []
    for name in scenes_to_fetch:
        local_path = output_dir / name
        if local_path.exists():
            skipped.append(name)
        else:
            to_fetch.append(name)

    if skipped:
        log.info(f'Skipping {len(skipped)} already-fetched scene(s).')

    if not to_fetch:
        log.info('Nothing new to fetch.')
        raise typer.Exit()

    log.info(f'{len(to_fetch)} scene(s) to fetch into {output_dir}.')
    if not Confirm.ask('Proceed?', default=True):
        log.info('Aborted.')
        raise typer.Exit()

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, scene_name in enumerate(to_fetch, 1):
        local_path = output_dir / scene_name
        log.info(f'[{i}/{len(to_fetch)}] Fetching {scene_name}...')
        pi.copy_scene(scene_name, local_path, verbose=verbose_transfer)
        log.info(f'  -> {local_path}')

    log.info(f'Done. Fetched {len(to_fetch)} scene(s) to {output_dir}.')

    log.info('Checking for GoPro SD card...')
    try:
        fetch_gopro(
            recordings_dir=output_dir,
            mount_point=None,
            threshold_ms=DEFAULT_THRESHOLD_MS,
            latest=False,
        )
    except typer.Exit:
        pass


@app.command()
def process_video(
    session_path: pathlib.Path = typer.Argument(
        ...,
        help='Path to a local session directory.',
    ),
    fps: float = typer.Option(
        10.0,
        help=('Framerate to use for the output video. Overridden by session metadata if present.'),
    ),
    output_name: str = typer.Option(
        FINGER_MP4,
        help='Output video filename (placed in the session directory).',
    ),
    include_audio: bool = typer.Option(
        True,
        help='Mux audio.wav into the output if present.',
    ),
):
    """Encode JPEG frames (and optionally audio) in a session directory into an MP4."""
    try:
        encode_session_video(session_path, fps, output_name, include_audio)
    except RuntimeError as e:
        log.error(str(e))
        raise typer.Exit(1)


@app.command(name='process-all')
def process_all(
    recordings_dir: pathlib.Path = typer.Option(
        DEFAULT_RECORDINGS_DIR,
        help='Directory containing session_* folders.',
    ),
    fps: float = typer.Option(
        10.0,
        help=('Framerate to use for output videos. Overridden by session metadata if present.'),
    ),
    output_name: str = typer.Option(
        FINGER_MP4,
        help='Output video filename to create in each session directory.',
    ),
    include_audio: bool = typer.Option(
        True,
        help='Mux audio.wav into each output if present.',
    ),
    force: bool = typer.Option(
        False,
        '--force',
        help='Reprocess sessions even when the output video already exists.',
    ),
):
    """Process all unprocessed sessions under recordings_dir."""
    recordings_dir = recordings_dir.resolve()
    if not recordings_dir.is_dir():
        log.error(f'Recordings directory not found: {recordings_dir}')
        raise typer.Exit(1)

    session_dirs = sorted(
        p
        for scene_dir in sorted(recordings_dir.iterdir())
        if scene_dir.is_dir() and scene_dir.name.startswith('scene_')
        for p in scene_dir.iterdir()
        if p.is_dir() and p.name.startswith('session_')
    )
    if not session_dirs:
        log.info(f'No scene_*/session_* directories found in {recordings_dir}')
        raise typer.Exit()

    to_process: list[pathlib.Path] = []
    already_processed: list[pathlib.Path] = []
    missing_video: list[pathlib.Path] = []
    for session_dir in session_dirs:
        if (session_dir / output_name).is_file():
            already_processed.append(session_dir)
            if not force:
                continue
        if not (session_dir / 'video').is_dir():
            missing_video.append(session_dir)
            continue
        to_process.append(session_dir)

    if already_processed:
        if force:
            log.info(f'Reprocessing {len(already_processed)} session(s) with existing outputs due to --force.')
        else:
            log.info(f'Skipping {len(already_processed)} already processed session(s).')
    if missing_video:
        log.warning(f'Skipping {len(missing_video)} session(s) without a video directory.')

    if not to_process:
        log.info('No unprocessed sessions found.')
        raise typer.Exit()

    log.info(f'Found {len(to_process)} unprocessed session(s) in {recordings_dir}.')
    if not Confirm.ask('Proceed?', default=True):
        log.info('Aborted.')
        raise typer.Exit()

    failures: list[tuple[pathlib.Path, str]] = []
    for i, session_dir in enumerate(to_process, 1):
        log.info(f'[{i}/{len(to_process)}] Processing {session_dir.name}...')
        try:
            encode_session_video(session_dir, fps, output_name, include_audio)
        except RuntimeError as e:
            failures.append((session_dir, str(e)))
            log.error(f'Failed {session_dir.name}: {e}')

    log.info(f'Completed. Success: {len(to_process) - len(failures)}, Failed: {len(failures)}.')
    if failures:
        raise typer.Exit(1)


@app.command(name='fetch-gopro')
def fetch_gopro(
    recordings_dir: pathlib.Path = typer.Option(
        DEFAULT_RECORDINGS_DIR,
        help='Directory containing session_* folders.',
    ),
    mount_point: pathlib.Path | None = typer.Option(
        None,
        help='GoPro SD card mount point. Auto-detected when omitted.',
    ),
    threshold_ms: float = typer.Option(
        DEFAULT_THRESHOLD_MS,
        help='Maximum allowed delta (ms) between gopro_sync_time and the inferred recording start.',
    ),
    latest: bool = typer.Option(
        False,
        '--latest',
        help='Only process the most recent session.',
    ),
):
    """Copy GoPro SD card footage into session directories that don't already have it."""
    recordings_dir = recordings_dir.resolve()
    if not recordings_dir.is_dir():
        log.error(f'Recordings directory not found: {recordings_dir}')
        raise typer.Exit(1)

    session_dirs = sorted(
        p
        for scene_dir in sorted(recordings_dir.iterdir())
        if scene_dir.is_dir() and scene_dir.name.startswith('scene_')
        for p in scene_dir.iterdir()
        if p.is_dir() and p.name.startswith('session_')
    )
    if not session_dirs:
        log.info(f'No scene_*/session_* directories found in {recordings_dir}')
        raise typer.Exit()

    if latest:
        session_dirs = [session_dirs[-1]]

    to_process: list[pathlib.Path] = []
    skipped_existing: list[str] = []
    skipped_no_sync: list[str] = []

    output_name = GOPRO_MP4
    for session_dir in session_dirs:
        if (session_dir / output_name).exists():
            skipped_existing.append(session_dir.name)
            continue
        try:
            session = SessionFiles.from_file(session_dir)
        except Exception as exc:
            log.warning(f'Could not load metadata for {session_dir.name}: {exc}')
            continue
        if session.metadata.gopro_sync_time is None:
            skipped_no_sync.append(session_dir.name)
            continue
        to_process.append(session_dir)

    if skipped_existing:
        log.info(f'Skipping {len(skipped_existing)} session(s) that already have {output_name}.')
    if skipped_no_sync:
        log.info(f'Skipping {len(skipped_no_sync)} session(s) with no gopro_sync_time: ' + ', '.join(skipped_no_sync))

    if not to_process:
        log.info('Nothing to do.')
        raise typer.Exit()

    log.info(f'{len(to_process)} session(s) to process.')

    failures: list[tuple[str, str]] = []
    for i, session_dir in enumerate(to_process, 1):
        session = SessionFiles.from_file(session_dir)
        sync_time = session.metadata.gopro_sync_time
        assert sync_time is not None  # filtered above
        log.info(f'[{i}/{len(to_process)}] {session_dir.name} (sync_time={sync_time.isoformat()})')
        try:
            src = find_gopro_video(
                start_time=sync_time,
                mount_point=mount_point,
                threshold_ms=threshold_ms,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            log.error(f'  Failed: {exc}')
            failures.append((session_dir.name, str(exc)))
            continue

        dst = session_dir / output_name
        shutil.copy2(src, dst)
        log.info(f'  -> {dst}')

    log.info(f'Done. Success: {len(to_process) - len(failures)}, Failed: {len(failures)}.')
    if failures:
        raise typer.Exit(1)


@app.command(name='inspect-zarr')
def inspect_zarr(
    scene_path: pathlib.Path = typer.Argument(
        ...,
        help='Scene directory containing scene.zarr, or a scene.zarr path directly.',
    ),
    save_frame: pathlib.Path | None = typer.Option(
        None,
        help='Save the first frame of episode_0 as a PNG to this path.',
    ),
):
    """Print the structure and metadata of a scene.zarr store."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    from polyumi_ingest.pzarr import PZarrInfo, inspect_pzarr, read_frame

    try:
        info: PZarrInfo = inspect_pzarr(scene_path)
    except FileNotFoundError as e:
        log.error(str(e))
        raise typer.Exit(1)

    console = Console()
    console.print(f'\n[bold]Store:[/bold] {info.zarr_path}')
    console.print(f'[bold]Format:[/bold] zarr v{info.zarr_format}\n')
    console.print('[bold]Tree:[/bold]')
    console.print(Text.from_ansi(str(info.tree)))
    console.print('\n[bold]Scene metadata:[/bold]')
    for k, v in sorted(info.attrs.items()):
        console.print(f'  {k}: {v}')

    for ep in info.episodes:
        duration = None
        if ep.episode_start is not None and ep.episode_end is not None:
            duration = ep.episode_end - ep.episode_start
            console.print(f'\n[bold]Episode {ep.index}[/bold] ({duration:.0f}s):')
        else:
            console.print(f'\n[bold]Episode {ep.index}:[/bold]')
        table = Table(show_header=True, header_style='bold cyan')
        table.add_column('Array')
        table.add_column('Shape')
        table.add_column('Info')
        if ep.finger_shape is not None:
            ts_info = ''
            if ep.finger_ts_range is not None:
                ts_info = f'{ep.finger_ts_range[0]:.3f} → {ep.finger_ts_range[1]:.3f} s'
                if ep.finger_ts_mean_delta_ms is not None:
                    ts_info += f'  (Δ={ep.finger_ts_mean_delta_ms:.1f} ms avg)'
            table.add_row('finger/frames', str(ep.finger_shape), ts_info)
        if ep.gopro_shape is not None:
            ts_info = ''
            if ep.gopro_ts_range is not None:
                ts_info = f'{ep.gopro_ts_range[0]:.3f} → {ep.gopro_ts_range[1]:.3f} s'
                if ep.gopro_ts_mean_delta_ms is not None:
                    ts_info += f'  (Δ={ep.gopro_ts_mean_delta_ms:.1f} ms avg)'
            table.add_row('gopro/frames', str(ep.gopro_shape), ts_info)
        if ep.audio_shape is not None:
            ts_info = ''
            if ep.audio_ts_range is not None:
                ts_info = f'{ep.audio_ts_range[0]:.3f} → {ep.audio_ts_range[1]:.3f} s'
            table.add_row('audio/data', str(ep.audio_shape), ts_info)
        if duration is not None:
            ep_info = f'{ep.episode_start:.3f} → {ep.episode_end:.3f} s  ({duration:.2f} s)'
            table.add_row('episode_start / end', '', ep_info)
        console.print(table)

    total_bytes = sum(f.stat().st_size for f in info.zarr_path.rglob('*') if f.is_file())
    console.print(f'\n[bold]Total size:[/bold] {_human_size(total_bytes)}')

    if save_frame is not None:
        from PIL import Image
        frame = read_frame(scene_path)
        Image.fromarray(frame).save(save_frame)
        console.print(f'\nSaved episode_0 frame 0 → {save_frame}')


@app.command(name='build-zarr')
def build_zarr(
    scene_path: pathlib.Path = typer.Argument(
        ...,
        help='Path to a processed scene directory containing session_* subdirectories.',
    ),
    skip_gopro: bool = typer.Option(
        False,
        '--skip-gopro',
        help='Skip GoPro frame ingestion (required until GoPro ingest is implemented).',
    ),
):
    """Build a pzarr working-format zarr store from a processed scene directory."""
    from polyumi_ingest.pzarr import build_pzarr

    try:
        zarr_path = build_pzarr(scene_path, skip_gopro=skip_gopro)
        log.info(f'Done. Zarr store written to {zarr_path}')
    except NotImplementedError as e:
        log.error(str(e))
        raise typer.Exit(1)
    except RuntimeError as e:
        log.error(str(e))
        raise typer.Exit(1)


if __name__ == '__main__':
    app()
