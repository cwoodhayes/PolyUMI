"""
Tests for the Pi->ROS timestamp contract at the ZMQ boundary.

The Pi stamps both streams in epoch nanoseconds at the capture instant, and this node stamps
message headers straight through. These pin that reinterpretation, the guard that catches a
stamp which does not line up with this host's clock, and the throttle that keeps the guard from
shouting once per message.
"""

import pytest

from polyumi_ros2.pi_receiver_node import WARN_INTERVAL_NS, LogThrottle, ns_to_ros_time, stamp_is_skewed

#: A plausible present-day capture instant, in epoch nanoseconds.
EPOCH_NS = 1_788_018_952_689_141_947

MAX_SKEW_S = 0.5


def test_ns_to_ros_time_splits_seconds_and_nanoseconds():
    """The split is exact — nanosecond detail survives, because it never goes through a float."""
    msg = ns_to_ros_time(EPOCH_NS)
    assert msg.sec == 1_788_018_952
    assert msg.nanosec == 689_141_947
    assert msg.sec * 1_000_000_000 + msg.nanosec == EPOCH_NS


def test_a_fresh_stamp_is_not_skewed():
    """A stamp within the transport delay is not worth a word."""
    assert not stamp_is_skewed(EPOCH_NS + 40_000_000, EPOCH_NS, MAX_SKEW_S)


def test_a_stamp_exactly_at_the_limit_is_not_skewed():
    """The threshold is inclusive, so a stamp sitting on it does not flap."""
    assert not stamp_is_skewed(EPOCH_NS + 500_000_000, EPOCH_NS, MAX_SKEW_S)


def test_a_device_time_counter_is_skewed():
    """A CLOCK_BOOTTIME counter published as if it were epoch reads as ~1970 and is caught."""
    uptime_ns = 3 * 3600 * 1_000_000_000  # a Pi up three hours
    assert stamp_is_skewed(EPOCH_NS, uptime_ns, MAX_SKEW_S)


def test_a_stamp_from_the_future_is_skewed():
    """Skew is caught in both directions — a Pi clock ahead of this host is just as broken."""
    assert stamp_is_skewed(EPOCH_NS, EPOCH_NS + 2_000_000_000, MAX_SKEW_S)


def test_repeat_offenders_are_throttled():
    """A mis-clocked Pi streaming at 10-50 Hz must not produce one line per message."""
    throttle = LogThrottle(WARN_INTERVAL_NS)
    emitted = [throttle.should_emit('skew:camera', EPOCH_NS + i) for i in range(50)]
    assert sum(emitted) == 1


def test_the_warning_repeats_once_the_interval_elapses():
    """Throttling suppresses the storm, not the problem — it comes back."""
    throttle = LogThrottle(WARN_INTERVAL_NS)
    assert throttle.should_emit('skew:camera', EPOCH_NS)
    assert not throttle.should_emit('skew:camera', EPOCH_NS + WARN_INTERVAL_NS - 1)
    assert throttle.should_emit('skew:camera', EPOCH_NS + WARN_INTERVAL_NS)


@pytest.mark.parametrize('other', ['skew:audio', 'idle:camera'])
def test_keys_throttle_independently(other):
    """Audio being late must not mask the camera being late, nor skew mask idle."""
    throttle = LogThrottle(WARN_INTERVAL_NS)
    assert throttle.should_emit('skew:camera', EPOCH_NS)
    assert throttle.should_emit(other, EPOCH_NS)
