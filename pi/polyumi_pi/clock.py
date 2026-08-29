"""
Clock-domain conversions for the Pi's outbound sensor streams.

Everything ``polyumi-pi`` puts on the wire is stamped in epoch nanoseconds (``CLOCK_REALTIME``)
at the instant the sample was *captured*, so the ROS side can stamp message headers straight
through with no conversion. That contract is written down on ``timestamp_ns`` in
``camera_frame.proto`` and ``audio_chunk.proto``; this module is the only place either stream
converts into it.

Two device clocks have to be mapped in:

* the camera's ``SensorTimestamp``, which libcamera reports on ``CLOCK_BOOTTIME``;
* the audio ADC instant, which PortAudio reports in its own arbitrary stream timebase.

Deliberately free of picamera2 and sounddevice imports, so it runs — and is tested — off the Pi.
"""

import time

# Above this, a measured audio buffering lag is not a lag but a bad clock reading.
MAX_PLAUSIBLE_LAG_S = 1.0

# EMA weight for new lag samples. Low: the physical buffering lag is near-constant, so most of
# the per-callback variation is sampling noise between stream.time and time.time_ns().
LAG_EMA_ALPHA = 0.1


def boottime_to_epoch_ns(boottime_ns: int) -> int:
    """
    Convert a ``CLOCK_BOOTTIME`` nanosecond timestamp to epoch nanoseconds.

    The offset is sampled on every call rather than cached at startup, and that is load-bearing:
    chrony steps ``CLOCK_REALTIME`` (at boot, and again on ``chronyc makestep``) without touching
    ``CLOCK_BOOTTIME``, so a cached offset silently goes wrong the moment the clock is stepped.
    Two ``clock_gettime`` calls per frame at 10 fps cost nothing.

    Uses the ``_ns`` variants because the float seconds returned by ``time.clock_gettime`` only
    resolve to about a microsecond at present-day epoch magnitudes.
    """
    offset_ns = time.clock_gettime_ns(time.CLOCK_REALTIME) - time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    return boottime_ns + offset_ns


class AudioStreamClock:
    """
    Map an audio callback onto the epoch instant its first sample hit the ADC.

    PortAudio reports the ADC instant in stream time, an arbitrary origin, so it is useless
    alone. What it gives us is the *lag* — how long ago the buffer was captured,
    ``stream.time - inputBufferAdcTime`` — which anchors against a ``time.time_ns()`` sampled in
    the same callback::

        capture_epoch_ns = epoch_ns - lag_ns

    The lag is tracked with an EMA rather than used raw: it is a near-constant property of the
    driver's buffering, so most of the per-callback variation is noise in when the two clocks
    were read. Re-anchoring on ``time.time_ns()`` every callback is also why nothing here has to
    worry about the sound card's clock drifting against ``CLOCK_REALTIME`` — only the lag is
    carried between callbacks, never an absolute origin.

    A callback whose ADC time is unusable reuses the tracked lag instead of switching to a
    different formula, so the timestamps stay continuous across the gap. If ADC time is *never*
    usable — some PortAudio backends do not fill it in — the clock stays in ``fixed`` mode and
    models the lag as one block, leaving the driver's own buffering to be absorbed by whatever
    ``latency.piezo_mic`` is eventually measured to be.

    Measured on the WM8960 HAT, 2026-08-29, 44.1 kHz / 20 ms blocks: ADC time is reported on
    every callback, and the true lag is 20.29-20.58 ms against the 20 ms one-block model. So on
    this hardware ``adc`` mode is what runs, and ``fixed`` would only be ~0.4 ms optimistic.
    """

    def __init__(self, blocksize: int, sample_rate: int) -> None:
        """
        Initialize.

        Parameters
        ----------
        blocksize:
            Frames per callback, used for the fixed-mode lag model.
        sample_rate:
            Stream sample rate in Hz.

        """
        if blocksize <= 0 or sample_rate <= 0:
            raise ValueError(f'blocksize and sample_rate must be positive, got {blocksize}, {sample_rate}')
        self._fixed_lag_s = blocksize / sample_rate
        self._lag_s: float | None = None
        self.n_unusable = 0

    @property
    def mode(self) -> str:
        """``'adc'`` once a usable ADC time has been seen, else ``'fixed'``."""
        return 'fixed' if self._lag_s is None else 'adc'

    @property
    def lag_s(self) -> float:
        """The lag currently being applied, in seconds."""
        return self._fixed_lag_s if self._lag_s is None else self._lag_s

    def stamp(self, stream_time_s: float, adc_time_s: float, epoch_ns: int) -> int:
        """
        Return the epoch-nanosecond capture instant of this callback's first sample.

        ``stream_time_s`` is PortAudio's current stream time, ``adc_time_s`` its
        ``inputBufferAdcTime``, and ``epoch_ns`` a ``time.time_ns()`` read in the same callback.
        """
        lag_s = stream_time_s - adc_time_s
        if adc_time_s > 0.0 and 0.0 <= lag_s <= MAX_PLAUSIBLE_LAG_S:
            if self._lag_s is None:
                self._lag_s = lag_s
            else:
                self._lag_s += LAG_EMA_ALPHA * (lag_s - self._lag_s)
        else:
            self.n_unusable += 1
        return epoch_ns - round(self.lag_s * 1e9)
