"""
pi_streamer.py - Runs on the Raspberry Pi Zero 2W.

Streams MJPEG frames over ZMQ to pi_receiver_node on the host PC.

Usage:
    python pi_streamer.py stream
    python pi_streamer.py stream --port 5555 --width 640 --height 480 --fps 10
"""

import asyncio
import logging
import multiprocessing
import os
import shutil
from multiprocessing.connection import Connection

import typer
import zmq
from audio_streamer import AudioStreamer
from cam_streamer import CameraStreamer
from led_manager import LEDManager
from rich.logging import RichHandler
from rich.prompt import Confirm

from polyumi_pi.files.session import DEFAULT_SESSION_BASE_DIR, SessionFiles
from polyumi_pi.gopro.gopro_wrapper import GoProWrapper

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
log = logging.getLogger('pi_streamer')

app = typer.Typer()


def _stop_child_process(process: multiprocessing.Process | None) -> None:
    if process is None or not process.is_alive():
        return

    process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        log.warning(f'Force-killing process pid={process.pid}')
        process.kill()
        process.join(timeout=2)


def _recv_child_stats(
    conn: Connection | None,
    name: str,
    timeout_s: float = 1.0,
) -> dict:
    """Receive one final stats payload from a child process."""
    if conn is None:
        return {}

    try:
        if not conn.poll(timeout_s):
            log.warning(f'No {name} stats received before timeout.')
            return {}
        payload = conn.recv()
        if not isinstance(payload, dict):
            log.warning(f'Unexpected {name} stats payload type: {type(payload)}')
            return {}
        return payload
    except (EOFError, OSError) as err:
        log.warning(f'Failed to receive {name} stats: {err}')
        return {}
    finally:
        conn.close()


def _run_video_streamer(
    port: int,
    fps: int,
    session: SessionFiles | None = None,
    stats_conn: Connection | None = None,
):
    context = zmq.Context()
    streamer = CameraStreamer(
        port=port,
        fps=fps,
        zmq_context=context,
        session=session,
        stats_conn=stats_conn,
    )
    try:
        streamer.start()
    finally:
        context.term()


def _run_audio_streamer(
    port: int,
    sample_rate: int,
    chunk_ms: int,
    channels: int,
    session: SessionFiles | None = None,
    stats_conn: Connection | None = None,
):
    context = zmq.Context()
    streamer = AudioStreamer(
        port=port,
        sample_rate=sample_rate,
        zmq_context=context,
        chunk_ms=chunk_ms,
        channels=channels,
        session=session,
        stats_conn=stats_conn,
    )
    try:
        streamer.start()
    finally:
        context.term()


@app.command()
def info():
    """Print camera information."""
    log.info(CameraStreamer.info())


@app.command()
def stream_video(
    port: int = typer.Option(5555, help='ZMQ PUSH port to bind on.'),
    fps: int = typer.Option(10, min=1, help='Target capture framerate (Hz).'),
):
    """Stream MJPEG frames over ZMQ."""
    log.info(f'Log level: {logging.getLevelName(log.level)}')
    context = zmq.Context()
    streamer = CameraStreamer(port=port, fps=fps, zmq_context=context)
    led = LEDManager()

    try:
        led.set_brightness(1.0)
        streamer.start()
    finally:
        context.term()
        led.set_brightness(0.0)


@app.command()
def stream_audio(
    port: int = typer.Option(5556, help='ZMQ PUSH port to bind on.'),
    sample_rate: int = typer.Option(44100, help='Audio sample rate (Hz).'),
    chunk_ms: int = typer.Option(20, help='Audio chunk size (ms).'),
    channels: int = typer.Option(2, help='Number of audio channels.'),
):
    """Stream audio data over ZMQ."""
    log.info(f'Log level: {logging.getLevelName(log.level)}')
    context = zmq.Context()
    streamer = AudioStreamer(
        port=port,
        sample_rate=sample_rate,
        zmq_context=context,
        chunk_ms=chunk_ms,
        channels=channels,
    )
    try:
        log.info('Starting audio streamer...')
        streamer.start()
    finally:
        context.term()


@app.command()
def stream(
    video_port: int = typer.Option(5555, help='ZMQ PUSH port for video.'),
    audio_port: int = typer.Option(5556, help='ZMQ PUSH port for audio.'),
    fps: int = typer.Option(10, min=1, help='Target capture framerate (Hz).'),
    sample_rate: int = typer.Option(16000, help='Audio sample rate (Hz).'),
    chunk_ms: int = typer.Option(20, help='Audio chunk size (ms).'),
    channels: int = typer.Option(1, help='Number of audio channels.'),
):
    """
    Stream both video and audio data over ZMQ.

    Intended for use on arm EE during inference.
    """
    log.info(f'Log level: {logging.getLevelName(log.level)}')
    led = LEDManager()
    cam_process: multiprocessing.Process | None = None
    audio_process: multiprocessing.Process | None = None

    try:
        led.set_brightness(1.0)
        log.info('Starting camera streamer...')
        cam_process = multiprocessing.Process(
            target=_run_video_streamer,
            args=(video_port, fps),
        )
        cam_process.start()

        log.info('Starting audio streamer...')
        audio_process = multiprocessing.Process(
            target=_run_audio_streamer,
            args=(audio_port, sample_rate, chunk_ms, channels),
        )
        audio_process.start()

        cam_process.join()
        audio_process.join()
    except KeyboardInterrupt:
        log.info('Keyboard interrupt received, stopping child streamers...')
    finally:
        _stop_child_process(cam_process)
        _stop_child_process(audio_process)
        led.set_brightness(0.0)


@app.command()
def record_episode(
    fps: int = typer.Option(10, min=1, help='Target capture framerate (Hz).'),
    sample_rate: int = typer.Option(16000, help='Audio sample rate (Hz).'),
    chunk_ms: int = typer.Option(20, help='Audio chunk size (ms).'),
    channels: int = typer.Option(1, help='Number of audio channels.'),
    robot: str = typer.Option(
        'polyumi_gripper', help='Name of the robot being recorded.'
    ),
    task: str = typer.Option('unspecified', help='Name of the task being recorded.'),
    gopro_identifier: str = typer.Option(
        ..., help='Last four digits of the GoPro serial number.'
    ),
):
    """
    Record an episode; video and audio data is routed to local files.

    Intended for use on PolyUMI gripper during data recording.
    """
    log.info(f'Log level: {logging.getLevelName(log.level)}')

    session = SessionFiles.create()
    session.metadata.robot = robot
    session.metadata.task = task

    log.info(f'Created session with ID {session.metadata.session_id} at {session.path}')
    session.init_audio(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=2,
        chunk_ms=chunk_ms,
    )
    session.init_video(
        fps=fps,
        width=CameraStreamer.CAPTURE_WIDTH,
        height=CameraStreamer.CAPTURE_HEIGHT,
    )
    led = LEDManager()

    async def _run() -> None:
        cam_process: multiprocessing.Process | None = None
        audio_process: multiprocessing.Process | None = None
        video_parent_conn: Connection | None = None
        audio_parent_conn: Connection | None = None

        try:
            async with GoProWrapper(gopro_identifier) as gopro:
                log.info('GoPro connected')
                sync_time = await gopro.set_timestamp()
                session.set_gopro_sync_time(sync_time)
                log.info(f'GoPro clock synced to {sync_time.isoformat()}')

                log.info('Starting GoPro recording...')
                await gopro.start_recording()

                try:
                    led_brightness = 1.0
                    session.metadata.led_brightness = led_brightness
                    led.set_brightness(led_brightness)

                    log.info('Starting camera streamer...')
                    video_parent_conn, video_child_conn = multiprocessing.Pipe(duplex=False)
                    cam_process = multiprocessing.Process(
                        target=_run_video_streamer,
                        args=(None, fps, session, video_child_conn),
                    )
                    cam_process.start()
                    video_child_conn.close()

                    log.info('Starting audio streamer...')
                    audio_parent_conn, audio_child_conn = multiprocessing.Pipe(duplex=False)
                    audio_process = multiprocessing.Process(
                        target=_run_audio_streamer,
                        args=(None, sample_rate, chunk_ms, channels, session, audio_child_conn),
                    )
                    audio_process.start()
                    audio_child_conn.close()

                    # Use to_thread so the event loop stays alive for BLE keepalives.
                    await asyncio.gather(
                        asyncio.to_thread(cam_process.join),
                        asyncio.to_thread(audio_process.join),
                    )

                    session.metadata.to_file()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    log.info('Keyboard interrupt received, stopping child streamers...')
                finally:
                    _stop_child_process(cam_process)
                    _stop_child_process(audio_process)

                    try:
                        await gopro.stop_recording()
                        log.info('GoPro recording stopped')
                    except BaseException as e:
                        log.warning(f'Failed to stop GoPro recording: {e}')
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info('Recording interrupted before streamers started.')
        except Exception as e:
            log.error(f'Unexpected error during recording: {e}', exc_info=True)
        finally:
            video_stats = _recv_child_stats(video_parent_conn, name='video')
            audio_stats = _recv_child_stats(audio_parent_conn, name='audio')

            if 'n_video_frames' in video_stats:
                session.metadata.n_video_frames = int(video_stats['n_video_frames'])
            if 'video_dropped_frames' in video_stats:
                session.metadata.video_dropped_frames = int(
                    video_stats['video_dropped_frames']
                )
            if 'n_audio_chunks' in audio_stats:
                session.metadata.n_audio_chunks = int(audio_stats['n_audio_chunks'])
            if 'audio_dropped_chunks' in audio_stats:
                session.metadata.audio_dropped_chunks = int(
                    audio_stats['audio_dropped_chunks']
                )
            if 'audio_start_time_ns' in audio_stats:
                session.metadata.audio_start_time_ns = audio_stats['audio_start_time_ns']

            led.set_brightness(0.0)
            session.finalize()
            log.info(
                f'Session finalized (t={session.metadata.duration_s}). '
                f'Data saved to {session.path}'
            )

    asyncio.run(_run())


@app.command()
def record_gopro(
    identifier: str = typer.Option(..., help='Last four digits of the GoPro serial number.'),
    duration: float = typer.Option(..., help='Recording duration in seconds.'),
    sync_clock: bool = typer.Option(True, help='Sync GoPro clock to system time before recording.'),
):
    """
    Trigger a timed GoPro recording over BLE.

    Connects to the GoPro, optionally syncs its clock, starts recording,
    waits for *duration* seconds, then stops recording.
    """
    log.info(f'Log level: {logging.getLevelName(log.level)}')

    async def _run() -> None:
        async with GoProWrapper(identifier) as gopro:
            log.info(f'Connected to GoPro {identifier}')
            if sync_clock:
                await gopro.set_timestamp()
                log.info('GoPro clock synced to system time')
            log.info(f'Starting GoPro recording for {duration}s...')
            await gopro.start_recording()
            await asyncio.sleep(duration)
            await gopro.stop_recording()
            log.info('GoPro recording stopped')

    asyncio.run(_run())


@app.command('clean-sessions')
def clean_sessions():
    """Delete all session recordings in the default session directory."""
    base_dir = DEFAULT_SESSION_BASE_DIR
    if not base_dir.exists():
        log.info(f'No session directory found at {base_dir}')
        return

    targets = list(base_dir.glob('session_*'))
    if not targets:
        log.info(f'No session entries found in {base_dir}')
        return

    if not Confirm.ask(
        f'Delete {len(targets)} session entries in {base_dir}?',
        default=False,
    ):
        log.info('Aborted cleaning sessions.')
        return

    removed = 0
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
            removed += 1
        elif path.is_file():
            path.unlink()
            removed += 1

    log.info(f'Removed {removed} session entries from {base_dir}')


if __name__ == '__main__':
    app()
