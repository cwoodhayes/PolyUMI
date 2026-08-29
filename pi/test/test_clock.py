"""Unit tests for the outbound stream clock conversions."""

import time

import pytest
from polyumi_pi.clock import LAG_EMA_ALPHA, MAX_PLAUSIBLE_LAG_S, AudioStreamClock, boottime_to_epoch_ns

BLOCKSIZE = 320
SAMPLE_RATE = 16000
FIXED_LAG_S = BLOCKSIZE / SAMPLE_RATE  # 0.02


def test_boottime_to_epoch_matches_a_live_reading():
    """A boottime reading converts to within a millisecond of the epoch clock."""
    boot_ns = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    epoch_ns = time.time_ns()
    assert abs(boottime_to_epoch_ns(boot_ns) - epoch_ns) < 1_000_000


def test_boottime_to_epoch_is_epoch_magnitude():
    """The result lands in present-day epoch nanoseconds, not seconds-since-boot."""
    converted = boottime_to_epoch_ns(time.clock_gettime_ns(time.CLOCK_BOOTTIME))
    # Anything before 2020 means the boottime offset was not applied.
    assert converted > 1_577_836_800_000_000_000


def test_boottime_offset_is_resampled_per_call(monkeypatch):
    """A stepped CLOCK_REALTIME is picked up immediately, not held from a cached offset."""
    readings = iter([1_000, 100, 5_000, 100])  # (realtime, boottime) pairs

    def fake(clk):
        return next(readings)

    monkeypatch.setattr(time, 'clock_gettime_ns', fake)
    assert boottime_to_epoch_ns(0) == 900
    assert boottime_to_epoch_ns(0) == 4900


def make_clock() -> AudioStreamClock:
    """Return a clock configured for the 20 ms / 16 kHz stream the Pi runs."""
    return AudioStreamClock(blocksize=BLOCKSIZE, sample_rate=SAMPLE_RATE)


def test_rejects_nonsense_configuration():
    """A zero or negative blocksize/sample rate is a programming error, not a fallback."""
    with pytest.raises(ValueError):
        AudioStreamClock(blocksize=0, sample_rate=SAMPLE_RATE)
    with pytest.raises(ValueError):
        AudioStreamClock(blocksize=BLOCKSIZE, sample_rate=0)


def test_starts_in_fixed_mode_with_one_block_of_lag():
    """Before any usable ADC time, the lag is modelled as a single block."""
    clock = make_clock()
    assert clock.mode == 'fixed'
    assert clock.lag_s == pytest.approx(FIXED_LAG_S)


def test_unfilled_adc_time_keeps_fixed_mode():
    """A backend that never fills in inputBufferAdcTime leaves the clock in fixed mode."""
    clock = make_clock()
    epoch_ns = 1_788_000_000_000_000_000
    for i in range(10):
        stamped = clock.stamp(stream_time_s=i * 0.02, adc_time_s=0.0, epoch_ns=epoch_ns)
        assert stamped == epoch_ns - round(FIXED_LAG_S * 1e9)
    assert clock.mode == 'fixed'
    assert clock.n_unusable == 10


def test_first_usable_sample_switches_to_adc_mode():
    """The first usable ADC time is adopted whole, with no EMA ramp from the fixed model."""
    clock = make_clock()
    epoch_ns = 1_788_000_000_000_000_000
    stamped = clock.stamp(stream_time_s=1.05, adc_time_s=1.0, epoch_ns=epoch_ns)
    assert clock.mode == 'adc'
    assert clock.lag_s == pytest.approx(0.05)
    assert stamped == epoch_ns - round(0.05 * 1e9)


def test_lag_is_smoothed_towards_new_samples():
    """Subsequent samples move the tracked lag by the EMA weight, not all the way."""
    clock = make_clock()
    clock.stamp(stream_time_s=1.05, adc_time_s=1.0, epoch_ns=0)
    clock.stamp(stream_time_s=2.07, adc_time_s=2.0, epoch_ns=0)
    assert clock.lag_s == pytest.approx(0.05 + LAG_EMA_ALPHA * (0.07 - 0.05))


def test_lag_converges_on_a_constant_lag():
    """Repeated identical samples drive the tracked lag to that value."""
    clock = make_clock()
    for i in range(500):
        clock.stamp(stream_time_s=i * 0.02 + 0.031, adc_time_s=i * 0.02, epoch_ns=0)
    assert clock.lag_s == pytest.approx(0.031, abs=1e-6)


def test_unusable_sample_holds_the_tracked_lag():
    """A bad reading reuses the tracked lag rather than jumping to the fixed model."""
    clock = make_clock()
    clock.stamp(stream_time_s=1.05, adc_time_s=1.0, epoch_ns=0)
    epoch_ns = 1_788_000_000_000_000_000

    held = clock.stamp(stream_time_s=2.0, adc_time_s=0.0, epoch_ns=epoch_ns)

    assert clock.mode == 'adc'
    assert clock.n_unusable == 1
    assert held == epoch_ns - round(0.05 * 1e9)


@pytest.mark.parametrize(
    'stream_time_s,adc_time_s',
    [
        (1.0, 2.0),  # negative lag: ADC time in the future
        (1.0 + MAX_PLAUSIBLE_LAG_S + 0.1, 1.0),  # implausibly large lag
        (1.0, 0.0),  # unfilled
        (1.0, -1.0),  # negative ADC time
    ],
)
def test_implausible_readings_are_rejected(stream_time_s, adc_time_s):
    """Readings outside the plausible band never enter the tracked lag."""
    clock = make_clock()
    clock.stamp(stream_time_s=stream_time_s, adc_time_s=adc_time_s, epoch_ns=0)
    assert clock.mode == 'fixed'
    assert clock.n_unusable == 1


def test_stamps_are_continuous_across_a_dropout():
    """Timestamps step by the callback interval across a run of unusable readings."""
    clock = make_clock()
    interval_ns = round(0.02 * 1e9)
    epoch0 = 1_788_000_000_000_000_000
    for i in range(20):  # settle the tracked lag
        clock.stamp(stream_time_s=i * 0.02 + 0.031, adc_time_s=i * 0.02, epoch_ns=epoch0 + i * interval_ns)

    good = clock.stamp(stream_time_s=20 * 0.02 + 0.031, adc_time_s=20 * 0.02, epoch_ns=epoch0 + 20 * interval_ns)
    dropped = clock.stamp(stream_time_s=21 * 0.02 + 0.031, adc_time_s=0.0, epoch_ns=epoch0 + 21 * interval_ns)

    assert dropped - good == pytest.approx(interval_ns, abs=1_000_000)
