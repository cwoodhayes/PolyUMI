"""
Tests for the Pi->ROS timestamp contract at the ZMQ boundary.

The Pi stamps both streams in epoch nanoseconds at the capture instant, and this node stamps
message headers straight through. That contract used to be asserted only in a docstring, while
the camera stream actually sent a CLOCK_BOOTTIME counter — every image header read as 1970, and
nothing anywhere noticed. These pin the reinterpretation and the guard that now notices.
"""

import pytest
import rclpy.time

from polyumi_ros2.pi_receiver_node import SKEW_WARN_INTERVAL_NS, PiReceiverNode, ns_to_ros_time

#: A plausible present-day capture instant, in epoch nanoseconds.
EPOCH_NS = 1_788_018_952_689_141_947


def test_ns_to_ros_time_splits_seconds_and_nanoseconds():
    """The split is exact, with no float rounding at epoch magnitudes."""
    msg = ns_to_ros_time(EPOCH_NS)
    assert msg.sec == 1_788_018_952
    assert msg.nanosec == 689_141_947
    assert msg.sec * 1_000_000_000 + msg.nanosec == EPOCH_NS


def test_ns_to_ros_time_keeps_sub_microsecond_detail():
    """Nanosecond detail survives; the conversion never goes through a float."""
    assert ns_to_ros_time(EPOCH_NS + 1).nanosec == 689_141_948


class _FakeClock:
    """Stands in for the node's clock so the tests do not race real time."""

    def __init__(self, now_ns: int):
        self.now_ns = now_ns

    def now(self) -> rclpy.time.Time:
        """Return the currently-set instant."""
        return rclpy.time.Time(nanoseconds=self.now_ns)


class _FakeLogger:
    """Collects the messages the guard emits."""

    def __init__(self):
        self.errors: list[str] = []

    def error(self, msg: str) -> None:
        """Record an error-level message."""
        self.errors.append(msg)


class _FakeNode:
    """The slice of PiReceiverNode that _warn_skew actually touches."""

    def __init__(self, now_ns: int, max_clock_skew_s: float = 0.5):
        self._clock = _FakeClock(now_ns)
        self._logger = _FakeLogger()
        self._max_clock_skew_s = max_clock_skew_s
        self._last_skew_warn: dict[str, rclpy.time.Time] = {}

    def get_clock(self) -> _FakeClock:
        """Return the fake clock."""
        return self._clock

    def get_logger(self) -> _FakeLogger:
        """Return the fake logger."""
        return self._logger


def warn_skew(node: _FakeNode, stream: str, stamp_ns: int) -> None:
    """Invoke the real guard against the fake node."""
    PiReceiverNode._warn_skew(node, stream, stamp_ns)


def test_a_fresh_stamp_is_silent():
    """A stamp within the transport delay is not worth a word."""
    node = _FakeNode(now_ns=EPOCH_NS + 40_000_000)  # 40 ms of transport
    warn_skew(node, 'camera', EPOCH_NS)
    assert node._logger.errors == []


def test_a_stamp_exactly_at_the_limit_is_silent():
    """The threshold is inclusive, so a stamp sitting on it does not flap."""
    node = _FakeNode(now_ns=EPOCH_NS + 500_000_000, max_clock_skew_s=0.5)
    warn_skew(node, 'camera', EPOCH_NS)
    assert node._logger.errors == []


def test_a_boottime_stamp_is_caught():
    """The original bug: a CLOCK_BOOTTIME counter published as if it were epoch."""
    uptime_ns = 3 * 3600 * 1_000_000_000  # a Pi up three hours
    node = _FakeNode(now_ns=EPOCH_NS)
    warn_skew(node, 'camera', uptime_ns)
    assert len(node._logger.errors) == 1
    assert 'camera' in node._logger.errors[0]
    assert 'pi-provisioning' in node._logger.errors[0]


def test_a_stamp_from_the_future_is_caught():
    """Skew is caught in both directions — a Pi clock ahead of this host is just as broken."""
    node = _FakeNode(now_ns=EPOCH_NS)
    warn_skew(node, 'audio', EPOCH_NS + 2_000_000_000)
    assert len(node._logger.errors) == 1


def test_repeat_offenders_are_throttled():
    """A mis-clocked Pi streaming at 10-50 Hz must not produce one error per message."""
    node = _FakeNode(now_ns=EPOCH_NS)
    for _ in range(50):
        warn_skew(node, 'camera', 12345)
    assert len(node._logger.errors) == 1


def test_the_warning_repeats_once_the_interval_elapses():
    """Throttling suppresses the storm, not the problem — it comes back."""
    node = _FakeNode(now_ns=EPOCH_NS)
    warn_skew(node, 'camera', 12345)
    node._clock.now_ns += SKEW_WARN_INTERVAL_NS
    warn_skew(node, 'camera', 12345)
    assert len(node._logger.errors) == 2


def test_the_two_streams_throttle_independently():
    """Audio being late must not mask the camera being late, or the reverse."""
    node = _FakeNode(now_ns=EPOCH_NS)
    warn_skew(node, 'camera', 12345)
    warn_skew(node, 'audio', 12345)
    assert len(node._logger.errors) == 2


@pytest.mark.parametrize('stream', ['camera', 'audio'])
def test_the_guard_never_rewrites_the_stamp(stream):
    """The guard reports; publishing the unchanged value is the caller's job and stays so."""
    node = _FakeNode(now_ns=EPOCH_NS)
    stamp_ns = 12345
    warn_skew(node, stream, stamp_ns)
    assert ns_to_ros_time(stamp_ns).nanosec == 12345
