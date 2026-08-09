"""
ingest/main.py - PolyUMI ingest scripts to deal with pi's file & build pzarr stores.

See docs/data-format.md for an overview of the pzarr format.
"""

import inspect
import json
import logging
import os
import pathlib
import shutil

import typer
from polyumi_pi.files.session import SessionFiles
from rich.logging import RichHandler
from rich.prompt import Confirm

from polyumi_ingest.export.dp import MIN_SEGMENT_STEPS
from polyumi_ingest.gopro_fetch import DEFAULT_THRESHOLD_MS, find_gopro_video
from polyumi_ingest.pi_fetch import PiFetch
from polyumi_ingest.preproc import (
    available_preprocessing_steps,
    run_preprocessing,
    run_preprocessing_on_recordings,
)
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
    except typer.Exit as exc:
        if exc.exit_code not in (None, 0):
            raise
        log.info('GoPro footage not copied — mount the SD card and run "pingest fetch-gopro" to add it.')


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
        help='Directory containing scene_* folders.',
    ),
    skip_gopro: bool = typer.Option(
        False,
        '--skip-gopro',
        help='Skip GoPro frame ingestion.',
    ),
    force: bool = typer.Option(
        False,
        '--force',
        help='Rebuild zarr stores even if they already exist.',
    ),
):
    """Build pzarr stores for all scenes under recordings_dir."""
    from polyumi_ingest.pzarr import build_pzarr

    recordings_dir = recordings_dir.resolve()
    if not recordings_dir.is_dir():
        log.error(f'Recordings directory not found: {recordings_dir}')
        raise typer.Exit(1)

    scene_dirs = sorted(p for p in recordings_dir.iterdir() if p.is_dir() and p.name.startswith('scene_'))
    if not scene_dirs:
        log.info(f'No scene_* directories found in {recordings_dir}')
        raise typer.Exit()

    to_process: list[pathlib.Path] = []
    skipped: list[pathlib.Path] = []
    for scene_dir in scene_dirs:
        if (scene_dir / 'scene.zarr').exists() and not force:
            skipped.append(scene_dir)
        else:
            to_process.append(scene_dir)

    if skipped:
        log.info(f'Skipping {len(skipped)} scene(s) with existing zarr stores.')

    if not to_process:
        log.info('Nothing to process.')
        raise typer.Exit()

    log.info(f'{len(to_process)} scene(s) to build.')
    if not Confirm.ask('Proceed?', default=True):
        log.info('Aborted.')
        raise typer.Exit()

    failures: list[tuple[pathlib.Path, str]] = []
    for i, scene_dir in enumerate(to_process, 1):
        log.info(f'[{i}/{len(to_process)}] Building {scene_dir.name}...')
        try:
            zarr_path = build_pzarr(scene_dir, skip_gopro=skip_gopro)
            log.info(f'  -> {zarr_path}')
        except Exception as e:
            # Anything at all, not just RuntimeError: a scene that can't be built shouldn't
            # abandon the scenes after it in the batch. Per-episode failures never reach here —
            # build_pzarr flags those and keeps going (see episode_status).
            failures.append((scene_dir, f'{type(e).__name__}: {e}'))
            log.error(f'  Failed: {e}')

    log.info(f'Done. Success: {len(to_process) - len(failures)}, Failed: {len(failures)}.')
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
    console.print(f'[bold]Format:[/bold] zarr v{info.zarr_format}, pzarr v{info.pzarr_version}\n')
    console.print('[bold]Tree:[/bold]')
    console.print(Text.from_ansi(str(info.tree)))
    console.print('\n[bold]Scene metadata:[/bold]')
    for k, v in sorted(info.attrs.items()):
        console.print(f'  {k}: {v}')

    def _fmt_rate(freq_hz: float | None) -> str:
        if freq_hz is None:
            return ''
        if freq_hz >= 1000:
            return f'{freq_hz / 1000:.1f} kHz'
        return f'{freq_hz:.2f} Hz'

    def _fmt_ts(ts_range: tuple[float, float] | None) -> str:
        if ts_range is None:
            return ''
        return f'{ts_range[0]:.3f} → {ts_range[1]:.3f} s'

    def _add_stream_row(table: Table, label: str, stream) -> None:
        if stream.shape is None:
            return
        table.add_row(label, str(stream.shape), _fmt_rate(stream.freq_hz), _fmt_ts(stream.ts_range))

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
        table.add_column('Rate', justify='right')
        table.add_column('Timestamps')
        _add_stream_row(table, 'finger/frames', ep.finger)
        _add_stream_row(table, 'finger/finger_piezo', ep.finger_piezo)
        _add_stream_row(table, 'finger/finger_air', ep.finger_air)
        _add_stream_row(table, 'gopro/frames', ep.gopro)
        _add_stream_row(table, 'gopro/accl', ep.gopro_accl)
        _add_stream_row(table, 'gopro/gyro', ep.gopro_gyro)
        _add_stream_row(table, 'gopro/gps', ep.gopro_gps)
        _add_stream_row(table, 'gopro/audio', ep.gopro_audio)
        if duration is not None:
            ep_info = f'{ep.episode_start:.3f} → {ep.episode_end:.3f} s  ({duration:.2f} s)'
            table.add_row('episode_start / end', '', '', ep_info)
        console.print(table)

    if info.optitrack is not None:
        ot = info.optitrack
        console.print('\n[bold]OptiTrack (scene-level):[/bold]')
        ot_table = Table(show_header=True, header_style='bold cyan')
        ot_table.add_column('Array')
        ot_table.add_column('Shape')
        ot_table.add_column('Rate', justify='right')
        ot_table.add_column('Timestamps')
        ot_table.add_row('optitrack/pose', str(ot.shape), _fmt_rate(ot.freq_hz), _fmt_ts(ot.ts_range))
        console.print(ot_table)

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
        help='Skip GoPro frame ingestion.',
    ),
):
    """Build a pzarr working-format zarr store from a processed scene directory."""
    from polyumi_ingest.pzarr import build_pzarr

    try:
        zarr_path = build_pzarr(scene_path, skip_gopro=skip_gopro)
        files = [f for f in zarr_path.rglob('*') if f.is_file()]
        src_size = sum(f.stat().st_size for f in files)
        log.info(f'Done. Zarr store written to {zarr_path} (total size: {_human_size(src_size)}).')
    except NotImplementedError as e:
        log.error(str(e))
        raise typer.Exit(1)
    except RuntimeError as e:
        log.error(str(e))
        raise typer.Exit(1)


@app.command(name='pp')
def preprocessing_pipeline(
    step: int | None = typer.Argument(
        None,
        min=1,
        help='Preprocessing step number. Omit to run every registered step in order.',
    ),
    scene: pathlib.Path | None = typer.Option(
        None,
        '--scene',
        help='Scene directory or scene.zarr path. Omit to run on every scene under recordings_dir.',
    ),
    recordings_dir: pathlib.Path = typer.Option(
        DEFAULT_RECORDINGS_DIR,
        help='Directory containing scene_* folders when --scene is omitted.',
    ),
    copy: bool = typer.Option(
        False,
        '--copy',
        help='Write the step output to scene_pp[step].zarr instead of mutating scene.zarr.',
    ),
    force: bool = typer.Option(
        False,
        '--force',
        '-f',
        help='Re-run a step even if it has already been marked complete.',
    ),
    skip_gopro: bool = typer.Option(
        False,
        '--skip-gopro',
        help='Skip GoPro frame ingestion when auto-building missing pzarr stores.',
    ),
    list_steps: bool = typer.Option(
        False,
        '--list',
        '-l',
        help='List the available preprocessing steps and exit, without touching any scene.',
    ),
):
    """Run a preprocessing step, or the full preprocessing pipeline, on scene zarr stores."""
    if list_steps:
        _print_preprocessing_steps()
        return

    # When no step is specified, auto-build scene.zarr for scenes that don't have one yet,
    # so `pingest pp` works end-to-end on a freshly fetched scene directory.
    auto_build = step is None
    try:
        if scene is not None:
            if auto_build and scene.suffix != '.zarr' and _pzarr_needs_build(scene):
                log.info(f'No usable scene.zarr at {scene}; building pzarr first...')
                if not skip_gopro:
                    _require_gopro_mp4s(scene)
                _build_pzarr(scene, skip_gopro)
            output = run_preprocessing(scene, step_number=step, copy=copy, force=force)
            log.info(f'Done. Output: {output}')
        else:
            if auto_build:
                recordings_dir_resolved = recordings_dir.resolve()
                if recordings_dir_resolved.is_dir():
                    for scene_dir in sorted(
                        p for p in recordings_dir_resolved.iterdir() if p.is_dir() and p.name.startswith('scene_')
                    ):
                        if not _pzarr_needs_build(scene_dir):
                            continue
                        log.info(f'No usable scene.zarr for {scene_dir.name}; building pzarr first...')
                        # One unbuildable scene (no gopro.mp4 yet, unreadable sessions) must not
                        # stop the batch — it just won't have a store for run_preprocessing to
                        # find below, which is already reported as "no scene.zarr found".
                        try:
                            if not skip_gopro:
                                _require_gopro_mp4s(scene_dir)
                            _build_pzarr(scene_dir, skip_gopro)
                        except Exception as e:
                            log.error(f'{scene_dir.name}: cannot build pzarr, skipping: {e}')
            outputs = run_preprocessing_on_recordings(recordings_dir, step_number=step, copy=copy, force=force)
            if outputs:
                log.info(f'Done. Processed {len(outputs)} scene(s).')
            else:
                log.info('No scenes processed.')
    except (FileNotFoundError, FileExistsError, KeyError) as e:
        log.exception(e)
        raise typer.Exit(1)


@app.command(name='calibrate-gripper')
def calibrate_gripper(
    scene: pathlib.Path = typer.Option(
        ...,
        '--scene',
        help='Scene directory containing scene.zarr, or a scene.zarr path directly.',
    ),
    session: str | None = typer.Option(
        None,
        '--session',
        help='Only use this episode key (e.g. episode_1). Default: pool every episode in the scene.',
    ),
):
    """
    Derive S_closed, the ArUco tag separation with the gripper fully closed.

    Record a scene in which the gripper is opened and closed fully several times in front of the
    GoPro, holding it shut for a few seconds each cycle, then:

        pingest pp 4 --scene <scene>        # detect the finger tags
        pingest calibrate-gripper --scene <scene>

    Reads the per-frame detections (``raw_widths_m``), NOT the resampled ``width_m`` series — the
    latter is hold-extrapolated onto the GoPro grid and would drag the extremes around. See
    polyumi_ingest.gripper_calib for why the output is a table rather than a single number.
    """
    import numpy as np
    import zarr

    from polyumi_ingest.episode_status import episode_keys
    from polyumi_ingest.gripper_calib import closed_width_stats, format_report
    from polyumi_ingest.pzarr.scene_files import SceneFiles

    scene_zarr = SceneFiles.resolve_zarr_path(scene)
    if not scene_zarr.exists():
        log.error(f'No scene.zarr found at {scene}. Run `pingest pp 4 --scene {scene}` first.')
        raise typer.Exit(1)

    # Read-only: a calibration read must never restamp provenance or create groups, unlike the
    # preprocessing steps' SceneContext.open (mode 'a').
    root = zarr.open_group(str(scene_zarr), mode='r')
    keys = [session] if session else episode_keys(root)

    pooled: list[np.ndarray] = []
    for key in keys:
        path = f'{key}/annotations/gripper_width'
        if path not in root:
            log.warning(f'{key}: no gripper_width annotation; run `pingest pp 4 --force` on this scene.')
            continue
        grp = root[path]
        widths = np.asarray(grp['raw_widths_m'][:], dtype=np.float64)  # type: ignore[index]
        rate = float(grp.attrs.get('detection_rate', float('nan')))  # type: ignore[union-attr]
        log.info(f'{key}: {widths.size} detections, {rate:.1%} of frames')
        pooled.append(widths)

    if not pooled:
        log.error('No gripper-width detections found in any episode.')
        raise typer.Exit(1)

    try:
        stats = closed_width_stats(np.concatenate(pooled))
    except ValueError as e:
        log.error(str(e))
        raise typer.Exit(1)

    print()
    print(format_report(stats))
    print()
    print('Put this in ingest/config/gripper_calib.yaml (the DP exporter reads it):')
    print('  gripper_fingers:')
    print(f'    closed_mm: {stats.s_closed_m * 1000:.2f}')
    print(f'    open_mm: {stats.max_m * 1000:.2f}')
    print()
    print('Then measure the FR3 side, which pairs with this to give gripper_offset_m:')
    print('  ros2 run polyumi_ros2 gripper_range_probe')


@app.command(name='archive-scene')
def archive_scene(
    scene_path: pathlib.Path = typer.Argument(
        ...,
        help='Scene directory containing scene.zarr, or a scene.zarr path directly.',
    ),
    output: pathlib.Path | None = typer.Option(
        None,
        help='Output path for the archive. Defaults to scene.zarr.zip inside the scene directory.',
    ),
    delete_zarr: bool = typer.Option(
        False,
        '--delete-zarr',
        help='Delete source scene.zarr after successful archiving.',
    ),
    force: bool = typer.Option(
        False,
        '--force',
        help='Overwrite an existing archive.',
    ),
):
    """
    Archive a scene to a self-contained zip for at-rest storage.

    Bundles scene.zarr together with each session's gopro.mp4 sidecar (the GoPro
    frames are decoded on demand from the mp4, not stored in the zarr) and the
    ORB-SLAM3 atlas if present. Paths are stored relative to the scene directory,
    so unzipping reproduces ``<scene>/scene.zarr`` + ``<scene>/session_*/gopro.mp4``
    — exactly the layout the frame reader resolves against.

    ZIP_STORED (no re-compress): zarr chunks are already Blosc-compressed and the
    mp4 is already an inter-frame codec, so deflate would only cost CPU.
    """
    import zipfile

    from polyumi_ingest.pzarr.scene_files import GOPRO_MP4, SceneFiles

    scene_path = scene_path.resolve()
    zarr_path = SceneFiles.resolve_zarr_path(scene_path)

    if not zarr_path.exists():
        log.error(f'No scene.zarr found at {scene_path}')
        raise typer.Exit(1)

    scene_dir = zarr_path.parent
    zip_path = output.resolve() if output else scene_dir / (zarr_path.name + '.zip')

    if zip_path.exists():
        if not force:
            log.error(f'Archive already exists: {zip_path}. Use --force to overwrite.')
            raise typer.Exit(1)
        zip_path.unlink()

    # scene.zarr contents + each session's gopro.mp4 + the atlas sidecar (if any).
    files = [f for f in zarr_path.rglob('*') if f.is_file()]
    gopro_mp4s = sorted(scene_dir.glob(f'session_*/{GOPRO_MP4}'))
    files.extend(gopro_mp4s)
    atlas = SceneFiles(path=scene_dir).orb_slam3_atlas
    if atlas.exists():
        files.append(atlas)

    src_size = sum(f.stat().st_size for f in files)
    log.info(f'Archiving {scene_dir.name} ({_human_size(src_size)}, {len(gopro_mp4s)} gopro.mp4) → {zip_path}')

    # Paths relative to the scene dir so the zip mirrors the on-disk scene layout.
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for file_path in files:
            zf.write(file_path, file_path.relative_to(scene_dir))

    zip_size = zip_path.stat().st_size
    log.info(f'Done. Archive: {_human_size(zip_size)} (source: {_human_size(src_size)})')

    if delete_zarr:
        if not Confirm.ask(f'Delete {zarr_path}?', default=False):
            raise typer.Exit()
        shutil.rmtree(zarr_path)
        log.info(f'Deleted {zarr_path}')


@app.command(name='export-mcap')
def export_mcap(
    scene_path: pathlib.Path = typer.Argument(
        ...,
        help='Scene directory containing scene.zarr, or a scene.zarr path directly.',
    ),
    output_dir: pathlib.Path | None = typer.Option(
        None,
        help='Directory to write .mcap files. Defaults to the scene directory.',
    ),
    episode: int | None = typer.Option(
        None,
        help='Export only this episode index. Omit to export all episodes.',
    ),
    jpeg_quality: int = typer.Option(
        85,
        help='JPEG re-encode quality for video frames (1–100).',
    ),
    audio_chunk_size: int = typer.Option(
        4096,
        min=1,
        help='Number of audio samples per RawAudio message.',
    ),
):
    """Export a pzarr scene to MCAP files for visualization in Foxglove."""
    from polyumi_ingest.export.mcap import export_scene_to_mcap

    try:
        written = export_scene_to_mcap(
            scene_path=scene_path,
            output_dir=output_dir,
            episode=episode,
            jpeg_quality=jpeg_quality,
            audio_chunk_size=audio_chunk_size,
        )
    except FileNotFoundError as e:
        log.error(str(e))
        raise typer.Exit(1)

    log.info(f'Exported {len(written)} episode(s):')
    for path in written:
        log.info(f'  {path}')


def _write_provenance_sidecar(output_path: pathlib.Path, provenance: list[dict]) -> pathlib.Path:
    """Write ``<output>.provenance.json`` beside a DP export, recording each episode's pose source."""
    sidecar_path = output_path.with_suffix(output_path.suffix + '.provenance.json')
    sidecar_path.write_text(json.dumps(provenance, indent=2))
    return sidecar_path


def _log_pose_source_summary(provenance: list[dict]) -> None:
    """Log a one-line-per-episode summary of which pose source each episode exported from."""
    for p in provenance:
        log.info(f'  {p["scene"]}/{p["episode"]}: pose={p["source"]} ({p["n_steps"]} steps)')


@app.command(name='export-dp')
def export_dp(
    scene_path: pathlib.Path = typer.Argument(
        ...,
        help='Scene directory containing scene.zarr, or a scene.zarr path directly.',
    ),
    output_path: pathlib.Path = typer.Option(
        ...,
        '--output',
        '-o',
        help='Output UMI ReplayBuffer path (a .zarr.zip file).',
    ),
    enforce_preprocessing: bool = typer.Option(
        True,
        '--enforce-preprocessing/--no-enforce-preprocessing',
        help='Require every preprocessing step to be complete before exporting. '
        'Disable to export a partially preprocessed scene; export can still fail if outputs '
        '(e.g. eef/pose_<source>) are missing, and the post-chirp start trim is applied independently '
        'whenever the chirp-end marker is present, regardless of this flag.',
    ),
    min_segment_steps: int = typer.Option(
        MIN_SEGMENT_STEPS,
        '--min-segment-steps',
        help='Shortest run of valid steps exported as its own episode. A session whose pose '
        'source drops out is split into the runs either side; runs shorter than this are '
        'discarded rather than emitted as episodes too short to sample a horizon from.',
    ),
):
    """
    Export a pzarr scene to a UMI-format ReplayBuffer (.zarr.zip).

    Poses come from eef/pose_<source>, written by preprocessing step 5 (eef-pose) for each
    source the scene has (optitrack and/or slam); run that first. This command then resolves
    which source each episode exports from — its eef.attrs['default_source'] (optitrack if
    present, else slam) unless overridden per-session in scene.json's pose_source_overrides.
    Frames are exported at the native GoPro rate; the training config sets the observation rate
    via obs_down_sample_steps. A per-episode pose-source provenance record is written to
    <output>.provenance.json and embedded in the .zarr.zip's meta attrs.
    """
    from polyumi_ingest.export.dp import export_scene_to_dp

    try:
        n, provenance = export_scene_to_dp(
            scene_path,
            output_path,
            enforce_preprocessing=enforce_preprocessing,
            min_segment_steps=min_segment_steps,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error(str(e))
        raise typer.Exit(1)

    _log_pose_source_summary(provenance)
    sidecar_path = _write_provenance_sidecar(output_path, provenance)
    log.info(f'Exported {n} episode(s) → {output_path} (provenance: {sidecar_path})')


@app.command(name='export-dataset')
def export_dataset(
    scene_paths: list[pathlib.Path] = typer.Argument(
        ...,
        help='Scene directories (each containing scene.zarr) to combine into one dataset.',
    ),
    output_path: pathlib.Path = typer.Option(
        ...,
        '--output',
        '-o',
        help='Output UMI ReplayBuffer path (a .zarr.zip file).',
    ),
    enforce_preprocessing: bool = typer.Option(
        True,
        '--enforce-preprocessing/--no-enforce-preprocessing',
        help='Require every preprocessing step to be complete on each scene before exporting. '
        'Disable to export partially preprocessed scenes; export can still fail if outputs '
        '(e.g. eef/pose_<source>) are missing, and the post-chirp start trim is applied independently '
        'whenever the chirp-end marker is present, regardless of this flag.',
    ),
    min_segment_steps: int = typer.Option(
        MIN_SEGMENT_STEPS,
        '--min-segment-steps',
        help='Shortest run of valid steps exported as its own episode. A session whose pose '
        'source drops out is split into the runs either side; runs shorter than this are '
        'discarded rather than emitted as episodes too short to sample a horizon from.',
    ),
):
    """
    Export EPISODE sessions from multiple pzarr scenes into a single UMI ReplayBuffer.

    Scenes are concatenated in the order given; episode_ends accumulates across all of them,
    so the result is indistinguishable from a single big scene to UmiDataset. Each scene needs
    preprocessing step 5 (eef-pose) run first, same as `export-dp`; the per-episode pose-source
    resolution (default vs. scene.json override) and provenance sidecar work the same way too.
    """
    from polyumi_ingest.export.dp import export_scenes_to_dp

    try:
        n, provenance = export_scenes_to_dp(
            scene_paths,
            output_path,
            enforce_preprocessing=enforce_preprocessing,
            min_segment_steps=min_segment_steps,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error(str(e))
        raise typer.Exit(1)

    _log_pose_source_summary(provenance)
    sidecar_path = _write_provenance_sidecar(output_path, provenance)
    log.info(f'Exported {n} episode(s) from {len(scene_paths)} scene(s) → {output_path} (provenance: {sidecar_path})')


def _step_summary(step_cls: type) -> str:
    """First line of a step class's docstring, with RST inline markup flattened for a terminal."""
    doc = inspect.getdoc(step_cls)
    if not doc:
        return '(no description)'
    return doc.splitlines()[0].replace('``', '')


def _print_preprocessing_steps() -> None:
    """Print the registered preprocessing steps, in execution order, with their summaries."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title='Preprocessing steps', title_justify='left', header_style='bold')
    table.add_column('#', justify='right')
    table.add_column('Name', style='bold')
    table.add_column('What it does')
    for step_cls in available_preprocessing_steps():
        table.add_row(str(step_cls.step_number), step_cls.step_name, _step_summary(step_cls))

    console = Console()
    console.print()
    console.print(table)
    console.print('\nRun one:  [bold]pingest pp <#> --scene <scene>[/bold]')
    console.print('Run all:  [bold]pingest pp --scene <scene>[/bold]\n')


def _build_pzarr(scene_dir: pathlib.Path, skip_gopro: bool) -> None:
    """Build pzarr for scene_dir, raising typer.Exit(1) on failure."""
    from polyumi_ingest.pzarr import build_pzarr

    zarr_path = scene_dir / 'scene.zarr'
    try:
        build_pzarr(scene_dir, skip_gopro=skip_gopro)
        log.info(f'  -> {zarr_path}')
    except (RuntimeError, NotImplementedError) as e:
        log.error(str(e))
        raise typer.Exit(1)


def _pzarr_needs_build(scene_dir: pathlib.Path) -> bool:
    """
    Report whether ``scene_dir`` still needs a scene.zarr built.

    A store whose ``build_complete`` attr is explicitly False was interrupted part-way and is
    missing episodes, so it gets rebuilt rather than preprocessed as if it were whole. Stores
    written before that attr existed don't have it at all, and a *missing* attr means "unknown,
    assume complete" — otherwise every pre-existing store would be rebuilt on sight.
    """
    import zarr

    zarr_path = scene_dir / 'scene.zarr'
    if not zarr_path.exists():
        return True
    try:
        root = zarr.open_group(str(zarr_path), mode='r')
    except Exception as e:
        log.warning(f'{scene_dir.name}: scene.zarr present but unreadable ({e}); rebuilding.')
        return True
    if root.attrs.get('build_complete') is False:
        log.warning(f'{scene_dir.name}: scene.zarr is from an interrupted build; rebuilding.')
        return True
    return False


def _require_gopro_mp4s(scene_dir: pathlib.Path) -> None:
    """Raise FileNotFoundError if any session under scene_dir is missing gopro.mp4."""
    from polyumi_ingest.pzarr.scene_files import GOPRO_MP4, SceneFiles

    scene = SceneFiles.from_path(scene_dir)
    missing = [s.path / GOPRO_MP4 for s in scene.sessions if not (s.path / GOPRO_MP4).exists()]
    if missing:
        joined = '\n  '.join(str(p) for p in missing)
        raise FileNotFoundError(
            f'Cannot auto-build pzarr for {scene_dir.name}: missing gopro.mp4 in '
            f'{len(missing)} session(s):\n  {joined}\n'
            f'Run `pingest fetch-gopro` first, or pass --skip-gopro to ingest without GoPro frames.'
        )


@app.command(name='debug-latest')
def debug_latest(
    host: str = typer.Option(DEFAULT_HOST, help='SSH hostname of the Pi.'),
    recordings_dir: pathlib.Path = typer.Option(
        DEFAULT_RECORDINGS_DIR,
        help='Local directory containing scene_* folders.',
    ),
    skip_gopro: bool = typer.Option(
        False,
        '--skip-gopro',
        help='Skip GoPro frame ingestion when building pzarr.',
    ),
    yes: bool = typer.Option(
        False,
        '--yes',
        '-y',
        help='Non-interactive: skip prompts and keep existing artifacts as-is.',
    ),
    run_pp: bool = typer.Option(
        False,
        '--pp',
        help='Run the full preprocessing pipeline after building pzarr, before MCAP export.',
    ),
    jpeg_quality: int = typer.Option(85, help='JPEG re-encode quality for MCAP export (1–100).'),
    audio_chunk_size: int = typer.Option(4096, min=1, help='Audio samples per RawAudio message.'),
):
    """
    Fetch the latest scene, build its pzarr, and export the last episode to MCAP.

    Useful for polyumi-pi development & testing the ingest pipeline quickly.
    """
    from polyumi_ingest.export.mcap import export_scene_to_mcap

    recordings_dir = recordings_dir.resolve()
    recordings_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: fetch latest scene from Pi
    pi = PiFetch(host)
    scene_name = pi.resolve_latest_scene()
    scene_dir = recordings_dir / scene_name
    if scene_dir.exists():
        log.info(f'Latest scene {scene_name} already fetched locally, skipping download.')
        if not skip_gopro:
            sessions_missing_gopro = [
                p
                for p in scene_dir.iterdir()
                if p.is_dir() and p.name.startswith('session_') and not (p / GOPRO_MP4).exists()
            ]
            if sessions_missing_gopro:
                log.info('GoPro video not yet present for some sessions, checking SD card...')
                try:
                    fetch_gopro(
                        recordings_dir=recordings_dir,
                        mount_point=None,
                        threshold_ms=DEFAULT_THRESHOLD_MS,
                        latest=False,
                    )
                except typer.Exit:
                    pass
    else:
        log.info(f'Fetching latest scene: {scene_name}...')
        pi.copy_scene(scene_name, scene_dir)
        log.info(f'  -> {scene_dir}')

    # Step 2: build pzarr
    zarr_path = scene_dir / 'scene.zarr'
    if zarr_path.exists():
        if yes:
            log.info(f'scene.zarr already exists for {scene_name}, skipping rebuild.')
        elif not Confirm.ask(f'scene.zarr already exists for {scene_name}. Rebuild?', default=False):
            log.info('Skipping pzarr rebuild.')
        else:
            _build_pzarr(scene_dir, skip_gopro)
    else:
        log.info(f'Building pzarr for {scene_name}...')
        _build_pzarr(scene_dir, skip_gopro)

    # Step 2.5: optionally run full preprocessing pipeline
    if run_pp:
        log.info('Running preprocessing pipeline...')
        try:
            output = run_preprocessing(scene_dir, step_number=None, copy=False, force=True)
            log.info(f'  -> {output}')
        except (FileNotFoundError, FileExistsError) as e:
            log.error(str(e))
            raise typer.Exit(1)

    # Step 3: export last episode to MCAP
    import zarr as _zarr

    _root = _zarr.open_group(str(zarr_path), mode='r')
    _ep_keys = sorted(k for k in _root.keys() if k.startswith('episode_'))
    if not _ep_keys:
        log.error(f'No episodes found in {zarr_path}')
        raise typer.Exit(1)
    last_ep = int(_ep_keys[-1].split('_')[-1])

    mcap_path = scene_dir / f'episode_{last_ep}.mcap'
    if mcap_path.exists():
        if yes:
            log.info(f'episode_{last_ep}.mcap already exists, skipping re-export.')
            raise typer.Exit()
        if not Confirm.ask(f'episode_{last_ep}.mcap already exists. Re-export?', default=False):
            log.info('Skipping MCAP export.')
            raise typer.Exit()
        mcap_path.unlink()

    log.info(f'Exporting episode {last_ep} to MCAP...')
    try:
        written = export_scene_to_mcap(
            scene_path=scene_dir,
            output_dir=scene_dir,
            episode=last_ep,
            jpeg_quality=jpeg_quality,
            audio_chunk_size=audio_chunk_size,
        )
    except FileNotFoundError as e:
        log.error(str(e))
        raise typer.Exit(1)

    for p in written:
        log.info(f'  -> {p}')


if __name__ == '__main__':
    app()
