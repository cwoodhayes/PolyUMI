"""Code for streaming audio data over zmq."""

import contextlib
import logging
import queue
import signal
import threading
import time

import sounddevice as sd
import zmq
from polyumi_pi_msgs.audio_chunk_pb2 import AudioChunk

from polyumi_pi.files.audio import AudioFile
from polyumi_pi.files.session import SessionFiles

log = logging.getLogger('pi_audio_streamer')


class AudioStreamer:
    """Class for streaming audio data over zmq."""

    DEVICE_NAME = 'wm8960-soundcard'  # matches ALSA card name

    def __init__(
        self,
        port: int | None,
        sample_rate: int,
        zmq_context: zmq.Context,
        chunk_ms: int = 20,
        channels: int = 2,
        session: SessionFiles | None = None,
    ):
        """
        Initialize the audio streamer.

        Args:
            port: ZMQ TCP port to publish audio chunks on, or None to disable
            sample_rate: Audio sample rate in Hz.
            zmq_context: Shared ZMQ context used to create the publisher socket
            chunk_ms: Audio callback chunk size in milliseconds.
            channels: Number of audio input channels to capture.
            session: Optional session object used for WAV recording metadata.

        """
        self.port = port
        self.sample_rate = sample_rate
        self.zmq_context = zmq_context
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.session = session

    @staticmethod
    def find_device_index(name: str) -> int:
        """Find sounddevice index by ALSA card name."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if (
                name.lower() in dev['name'].lower()
                and dev['max_input_channels'] > 0
            ):
                return i
        raise RuntimeError(
            f"Could not find input device matching '{name}'. "
            f'Available devices:\n{sd.query_devices()}'
        )

    @staticmethod
    def build_chunk(
        pcm_bytes: bytes, sample_rate: int, channels: int, timestamp_ns: int
    ) -> bytes:
        """Serialize an AudioChunk protobuf message."""
        chunk = AudioChunk()
        chunk.timestamp_ns = timestamp_ns
        chunk.pcm_data = pcm_bytes
        chunk.sample_rate = sample_rate
        chunk.channels = channels
        chunk.bit_depth = 16
        return chunk.SerializeToString()

    def start(self) -> None:
        """Record and stream audio data."""
        # Number of frames per callback chunk
        blocksize = int(self.sample_rate * self.chunk_ms / 1000)

        streaming_enabled = self.port is not None
        if self.session is not None and self.session.audio is not None:
            log.info(
                f'Audio will be recorded to {self.session.audio.path} '
                f'at {self.sample_rate} Hz, {self.channels} channels, '
                f'{blocksize} frames/chunk.'
            )
        elif not streaming_enabled:
            log.warning(
                'Audio streaming and recording are both disabled. No audio will'
                ' be captured.'
            )

        # ZMQ setup
        sock = None
        if streaming_enabled:
            sock = self.zmq_context.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, 200)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(f'tcp://*:{self.port}')
            log.info(f'PUSH AudioChunk on tcp://*:{self.port}')
        else:
            log.info('ZMQ audio streaming disabled (port is None).')

        # Small queue to decouple sounddevice callback from ZMQ publish
        audio_queue: queue.Queue = queue.Queue(maxsize=100)
        callback_drops = 0
        sent_chunks = 0
        last_stats = time.monotonic()
        stop_event = threading.Event()

        def handle_shutdown(signum, _frame):
            signal_name = signal.Signals(signum).name
            log.info(f'Received signal {signal_name}, shutting down.')
            stop_event.set()

        prev_sigint = signal.signal(signal.SIGINT, handle_shutdown)
        prev_sigterm = signal.signal(signal.SIGTERM, handle_shutdown)

        def callback(
            indata,
            frames: int,
            time_info,
            status: sd.CallbackFlags,
        ):
            nonlocal callback_drops
            if status:
                log.warning(f'[sounddevice] {status}')
            ts = time.time_ns()
            pcm_bytes = bytes(indata)
            if streaming_enabled:
                try:
                    audio_queue.put_nowait((pcm_bytes, ts))
                except queue.Full:
                    callback_drops += 1
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        audio_queue.put_nowait((pcm_bytes, ts))
                    except queue.Full:
                        callback_drops += 1

            # write to the file if enabled
            if (
                wav_writer is not None
                and self.session is not None
                and self.session.audio is not None
            ):
                wav_writer.writeframes(pcm_bytes)
                if self.session.metadata.audio_start_time_ns is None:
                    self.session.metadata.audio_start_time_ns = ts

        device_index = self.find_device_index(self.DEVICE_NAME)
        log.info(
            f'Using device index {device_index}: '
            f'{sd.query_devices(device_index)["name"]}'
        )
        log.info(
            f'Sample rate: {self.sample_rate} Hz '
            f'| Channels: {self.channels} | '
            f'Chunk: {self.chunk_ms}ms ({blocksize} frames)'
        )

        # Publisher thread - keeps ZMQ sends off the audio callback
        def publisher(socket: zmq.Socket):
            nonlocal callback_drops, sent_chunks, last_stats
            while not stop_event.is_set() or not audio_queue.empty():
                try:
                    pcm_bytes, ts = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                msg = self.build_chunk(
                    pcm_bytes, self.sample_rate, self.channels, ts
                )
                socket.send(msg)
                sent_chunks += 1

                now = time.monotonic()
                if now - last_stats >= 1.0:
                    log.info(
                        'Audio tx stats: '
                        f'sent={sent_chunks}/s '
                        f'queue={audio_queue.qsize()} '
                        f'cb_drops={callback_drops}'
                    )
                    sent_chunks = 0
                    callback_drops = 0
                    last_stats = now

        pub_thread = None
        if streaming_enabled:
            pub_thread = threading.Thread(
                target=publisher, args=(sock,), daemon=True
            )
            pub_thread.start()

        wav_writer = None
        # Start capture stream
        with contextlib.ExitStack() as stack:
            if self.session is not None and self.session.audio is not None:
                wav_writer = stack.enter_context(
                    self.session.audio.recording()
                )
            with sd.RawInputStream(
                device=device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                blocksize=blocksize,
                callback=callback,
            ):
                log.info('Streaming... Ctrl+C to stop.')
                try:
                    while not stop_event.is_set():
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    log.info('\nStopping.')
                    stop_event.set()

        stop_event.set()
        if pub_thread is not None:
            pub_thread.join(timeout=2.0)
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        if sock is not None:
            sock.close()
