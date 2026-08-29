"""Unit tests for the outbound audio stream's capture-instant timestamps."""

import pytest
from polyumi_pi.clock import LAG_EMA_ALPHA, MAX_PLAUSIBLE_LAG_S, AudioStreamClock

BLOCKSIZE = 320
SAMPLE_RATE = 16000
BLOCK_LAG_S = BLOCKSIZE / SAMPLE_RATE  # 0.02
EPOCH_NS = 1_788_000_000_000_000_000
#: PortAudio stream time has an arbitrary origin, but never a small one on the Pi's ALSA
#: backend — it is the ALSA hardware timestamp, so it reads in the thousands of seconds.
STREAM_T0 = 4891.0


def make_clock() -> AudioStreamClock:
    """Return a clock configured for the 20 ms / 16 kHz stream the Pi runs."""
    return AudioStreamClock(blocksize=BLOCKSIZE, sample_rate=SAMPLE_RATE)


def test_rejects_nonsense_configuration():
    """A zero or negative blocksize/sample rate is a programming error, not a fallback."""
    with pytest.raises(ValueError):
        AudioStreamClock(blocksize=0, sample_rate=SAMPLE_RATE)
    with pytest.raises(ValueError):
        AudioStreamClock(blocksize=BLOCKSIZE, sample_rate=0)


def test_starts_with_one_block_of_lag():
    """Before any usable reading, the lag is modelled as a single block."""
    assert make_clock().lag_s == pytest.approx(BLOCK_LAG_S)


def test_first_usable_sample_is_adopted_whole():
    """The first usable reading is taken as-is, with no EMA ramp from the block model."""
    clock = make_clock()
    stamped = clock.stamp(current_time_s=STREAM_T0 + 0.05, adc_time_s=STREAM_T0, epoch_ns=EPOCH_NS)
    assert clock.lag_s == pytest.approx(0.05)
    assert stamped == EPOCH_NS - round(0.05 * 1e9)


def test_lag_is_smoothed_towards_new_samples():
    """Subsequent samples move the tracked lag by the EMA weight, not all the way."""
    clock = make_clock()
    clock.stamp(current_time_s=STREAM_T0 + 0.05, adc_time_s=STREAM_T0, epoch_ns=0)
    clock.stamp(current_time_s=STREAM_T0 + 1.07, adc_time_s=STREAM_T0 + 1.0, epoch_ns=0)
    assert clock.lag_s == pytest.approx(0.05 + LAG_EMA_ALPHA * (0.07 - 0.05))


@pytest.mark.parametrize(
    'current_time_s,adc_time_s',
    [
        (STREAM_T0, STREAM_T0 + 1.0),  # negative lag: ADC time in the future
        (STREAM_T0 + MAX_PLAUSIBLE_LAG_S + 0.1, STREAM_T0),  # implausibly large lag
        (STREAM_T0, 0.0),  # ADC time unfilled
        (0.0, 0.0),  # neither reading filled — a zero lag is not a real one
        (STREAM_T0, -1.0),  # negative ADC time
    ],
)
def test_implausible_readings_are_rejected(current_time_s, adc_time_s):
    """Readings outside the plausible band never enter the tracked lag."""
    clock = make_clock()
    stamped = clock.stamp(current_time_s=current_time_s, adc_time_s=adc_time_s, epoch_ns=EPOCH_NS)
    assert clock.n_rejected == 1
    assert stamped == EPOCH_NS - round(BLOCK_LAG_S * 1e9)


def test_a_backend_that_never_fills_time_info_stays_on_the_block_model():
    """Every stamp falls back to one block, and the rejections are counted for the stats line."""
    clock = make_clock()
    for i in range(10):
        assert clock.stamp(current_time_s=STREAM_T0 + i * 0.02, adc_time_s=0.0, epoch_ns=EPOCH_NS) == (
            EPOCH_NS - round(BLOCK_LAG_S * 1e9)
        )
    assert clock.n_rejected == 10


def test_rejected_sample_holds_the_tracked_lag():
    """A bad reading reuses the tracked lag rather than snapping back to the block model."""
    clock = make_clock()
    clock.stamp(current_time_s=STREAM_T0 + 0.05, adc_time_s=STREAM_T0, epoch_ns=0)

    held = clock.stamp(current_time_s=STREAM_T0 + 1.0, adc_time_s=0.0, epoch_ns=EPOCH_NS)

    assert clock.n_rejected == 1
    assert held == EPOCH_NS - round(0.05 * 1e9)


def test_stamps_are_continuous_across_a_dropout():
    """Timestamps step by the callback interval across a run of unusable readings."""
    clock = make_clock()
    interval_ns = round(0.02 * 1e9)

    def adc(i: float) -> float:
        return STREAM_T0 + i * 0.02

    for i in range(20):  # settle the tracked lag
        clock.stamp(current_time_s=adc(i) + 0.031, adc_time_s=adc(i), epoch_ns=EPOCH_NS + i * interval_ns)

    good = clock.stamp(current_time_s=adc(20) + 0.031, adc_time_s=adc(20), epoch_ns=EPOCH_NS + 20 * interval_ns)
    dropped = clock.stamp(current_time_s=adc(21) + 0.031, adc_time_s=0.0, epoch_ns=EPOCH_NS + 21 * interval_ns)

    assert dropped - good == pytest.approx(interval_ns, abs=1_000_000)
