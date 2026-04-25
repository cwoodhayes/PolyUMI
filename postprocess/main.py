"""
postprocess/main.py - PolyUMI postprocessing scripts.

Usage:
    python main.py fetch --host conorpi
    python main.py fetch --host conorpi --latest
    python main.py process-video recordings/session_2024-01-01_12-00-00
"""

import logging
import os
import pathlib
import shutil

import typer
from gopro_fetch import DEFAULT_THRESHOLD_MS, find_gopro_video
from pi_fetch import PiFetch
from polyumi_pi.files.session import SessionFiles
from rich.logging import RichHandler
from rich.prompt import Confirm
from video_helpers import encode_session_video

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
log = logging.getLogger('postprocess')

app = typer.Typer()

DEFAULT_HOST = 'pi@polyumi-pi.local'
DEFAULT_RECORDINGS_DIR = pathlib.Path('recordings')
VIDEO_OUTPUT_NAME = 'finger.mp4'


@app.command()
def fetch(
    host: str = typer.Option(DEFAULT_HOST, help='SSH hostname of the Pi.'),
    output_dir: pathlib.Path = typer.Option(
        DEFAULT_RECORDINGS_DIR,
        help='Local directory to write sessions into.',
    ),
    latest: bool = typer.Option(
        False,
        '--latest',
        help='Only fetch the latest session.',
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
        session_name = pi.resolve_latest_session()
        sessions_to_fetch = [session_name]
        log.info(f'Latest session: {session_name}')
    else:
        log.info(f'Listing sessions on {host}...')
        sessions_to_fetch = pi.list_remote_sessions()
        log.info(f'Found {len(sessions_to_fetch)} session(s) on {host}.')

    if not sessions_to_fetch:
        log.info('No sessions to fetch.')
        raise typer.Exit()

    # filter out already-fetched sessions
    to_fetch = []
    skipped = []
    for name in sessions_to_fetch:
        local_path = output_dir / name
        if local_path.exists():
            skipped.append(name)
        else:
            to_fetch.append(name)

    if skipped:
        log.info(f'Skipping {len(skipped)} already-fetched session(s).')

    if not to_fetch:
        log.info('Nothing new to fetch.')
        raise typer.Exit()

    log.info(f'{len(to_fetch)} session(s) to fetch into {output_dir}.')
    if not Confirm.ask('Proceed?', default=True):
        log.info('Aborted.')
        raise typer.Exit()

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, session_name in enumerate(to_fetch, 1):
        local_path = output_dir / session_name
        log.info(f'[{i}/{len(to_fetch)}] Fetching {session_name}...')
        pi.copy_session(session_name, local_path, verbose=verbose_transfer)
        log.info(f'  -> {local_path}')

    log.info(f'Done. Fetched {len(to_fetch)} session(s) to {output_dir}.')


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
        VIDEO_OUTPUT_NAME,
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
        VIDEO_OUTPUT_NAME,
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

    session_dirs = sorted(p for p in recordings_dir.iterdir() if p.is_dir() and p.name.startswith('session_'))
    if not session_dirs:
        log.info(f'No session_* directories found in {recordings_dir}')
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
    output_name: str = typer.Option(
        'gopro.mp4',
        help='Filename to copy the GoPro video as inside each session directory.',
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

    session_dirs = sorted(p for p in recordings_dir.iterdir() if p.is_dir() and p.name.startswith('session_'))
    if not session_dirs:
        log.info(f'No session_* directories found in {recordings_dir}')
        raise typer.Exit()

    if latest:
        session_dirs = [session_dirs[-1]]

    to_process: list[pathlib.Path] = []
    skipped_existing: list[str] = []
    skipped_no_sync: list[str] = []

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


if __name__ == '__main__':
    app()
