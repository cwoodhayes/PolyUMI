"""Tests for the child->parent stats pipe in :mod:`polyumi_pi.main`."""

import multiprocessing
import sys
from unittest.mock import MagicMock

# polyumi_pi.main pulls in the Pi-only camera/GPIO stack at import time, and none of it is
# involved in draining a stats pipe. Stub it so this runs on a laptop as well as on the Pi.
for _name in ('libcamera', 'picamera2', 'lgpio', 'gpiozero', 'sounddevice'):
    sys.modules.setdefault(_name, MagicMock())

from polyumi_pi.main import _recv_child_stats  # noqa: E402


def test_merges_every_payload_and_lets_the_last_win():
    """
    An early payload survives even when the shutdown tally never arrives.

    That is the whole point: the camera child reports first_frame_metadata as soon as it
    has it, because a child killed at shutdown never sends its final payload and ingest
    cannot rebuild that anchor from the JPEGs on disk.
    """
    parent, child = multiprocessing.Pipe(duplex=False)
    child.send({'first_frame_metadata': {'FrameWallClock': 42}})
    child.send({'n_video_frames': 7, 'first_frame_metadata': {'FrameWallClock': 42}})
    child.close()

    assert _recv_child_stats(parent, name='video') == {
        'first_frame_metadata': {'FrameWallClock': 42},
        'n_video_frames': 7,
    }


def test_early_payload_survives_a_child_that_never_reports_its_tally():
    """A single early payload is returned when the child dies before its shutdown tally."""
    parent, child = multiprocessing.Pipe(duplex=False)
    child.send({'audio_start_time_ns': 123})
    child.close()

    assert _recv_child_stats(parent, name='audio') == {'audio_start_time_ns': 123}


def test_silent_child_yields_nothing():
    """A child that sent nothing gives an empty dict, not a hang."""
    parent, child = multiprocessing.Pipe(duplex=False)
    child.close()

    assert _recv_child_stats(parent, name='video', timeout_s=0.05) == {}
