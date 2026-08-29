"""
Capture-instant timestamps for the Pi's outbound audio stream.

Everything ``polyumi-pi`` puts on the wire is stamped in epoch nanoseconds (``CLOCK_REALTIME``)
at the instant the sample was *captured*, so the ROS side can stamp message headers straight
through with no conversion. That contract is written down on ``timestamp_ns`` in
``camera_frame.proto`` and ``audio_chunk.proto``. The camera reaches it for free — libcamera
stamps every frame with ``FrameWallClock``, already in that domain — so only audio needs work.

Deliberately free of sounddevice imports, so it runs — and is tested — off the Pi.
"""

# Above this, a measured buffering lag is not a lag but a bad clock reading.
MAX_PLAUSIBLE_LAG_S = 1.0


class AudioStreamClock:
    """
    Map an audio callback onto the epoch instant its first sample was captured.

    PortAudio reports the ADC instant in stream time, an arbitrary origin, so it is useless
    alone. What it gives us is the *lag* — how long ago the block was captured — which anchors
    against a ``time.time_ns()`` sampled in the same callback::

        lag_s = current_time_s - adc_time_s
        capture_epoch_ns = epoch_ns - lag_s * 1e9

    **Both readings must come from the callback's ``time_info``.** PortAudio's ALSA backend
    derives ``inputBufferAdcTime`` from ``currentTime`` by subtracting the capture buffer's
    occupancy, so their difference is the lag exactly. Measuring against ``stream.time`` instead
    re-reads the clock ~0.4 ms later and folds the callback's own startup cost into every
    timestamp.

    Each usable reading is applied as measured: it is the age of *this* callback's first sample,
    so a callback that ran late has genuinely been holding a fuller buffer and must be stamped
    with the larger lag. A reading outside the plausible band reuses the last good lag, so
    timestamps stay continuous across the gap; before any usable reading the lag is modelled as
    one block. Measured on the WM8960 HAT, 2026-08-29, 44.1 kHz / 20 ms blocks: 20.0027 ms, stdev
    0.011 ms. The one-block model is not a degraded fallback there, it is the same answer to
    within the two-frame quantisation of the ALSA delay.

    The lag is the capture buffer's occupancy, not the codec's own conversion delay — that
    remains for ``latency.piezo_mic`` to cover.
    """

    def __init__(self, blocksize: int, sample_rate: int) -> None:
        """
        Initialize.

        Parameters
        ----------
        blocksize:
            Frames per callback, used to model the lag until a usable reading arrives.
        sample_rate:
            Stream sample rate in Hz.

        """
        if blocksize <= 0 or sample_rate <= 0:
            raise ValueError(f'blocksize and sample_rate must be positive, got {blocksize}, {sample_rate}')
        self._block_lag_s = blocksize / sample_rate
        self._lag_s: float | None = None
        self.n_rejected = 0

    @property
    def lag_s(self) -> float:
        """The lag currently being applied, in seconds."""
        return self._block_lag_s if self._lag_s is None else self._lag_s

    def stamp(self, current_time_s: float, adc_time_s: float, epoch_ns: int) -> int:
        """
        Return the epoch-nanosecond capture instant of this callback's first sample.

        ``current_time_s`` and ``adc_time_s`` are the callback's ``time_info.currentTime`` and
        ``time_info.inputBufferAdcTime``; ``epoch_ns`` is a ``time.time_ns()`` read in the same
        callback. A lag of exactly zero is rejected along with the implausible ones — no capture
        buffer delivers a block it has not yet buffered, so it means the backend filled in
        neither reading.
        """
        lag_s = current_time_s - adc_time_s
        if 0.0 < lag_s <= MAX_PLAUSIBLE_LAG_S:
            self._lag_s = lag_s
        else:
            self.n_rejected += 1
        return epoch_ns - round(self.lag_s * 1e9)
