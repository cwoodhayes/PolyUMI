"""Tests for the shared GoPro-to-finger clock helper."""

import pathlib

import numpy as np
import pytest
import zarr
from polyumi_ingest.timebase import gopro_ts_in_finger_clock

GOPRO_TS = np.array([10.0, 11.0, 12.0], dtype=np.float64)


def _episode(tmp_path: pathlib.Path, *, with_time_sync_group: bool, offset_s: float | None) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / 'scene.zarr'), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ep.create_group('timestamps').create_array('gopro', data=GOPRO_TS)
    if with_time_sync_group:
        sync = ep.require_group('annotations').require_group('time_sync')
        if offset_s is not None:
            sync.attrs['gopro_to_finger_offset_s'] = offset_s
    return ep


def test_offset_is_subtracted_when_present(tmp_path: pathlib.Path) -> None:
    """The common case: a chirp offset shifts every GoPro timestamp into the finger clock."""
    ep = _episode(tmp_path, with_time_sync_group=True, offset_s=0.5)
    ts = gopro_ts_in_finger_clock(ep, require_offset=True)
    assert np.array_equal(ts, GOPRO_TS - 0.5)


def test_missing_time_sync_group_is_fatal_when_required(tmp_path: pathlib.Path) -> None:
    """No annotations/time_sync at all must refuse a sample-exact caller, not default to 0."""
    ep = _episode(tmp_path, with_time_sync_group=False, offset_s=None)
    with pytest.raises(RuntimeError, match='time_sync'):
        gopro_ts_in_finger_clock(ep, require_offset=True)


def test_missing_offset_attr_is_fatal_when_required(tmp_path: pathlib.Path) -> None:
    """
    A time_sync group with no offset attr must refuse, not silently use 0.0.

    An offset of exactly 0.0 stored on disk and a group with nothing stored are
    indistinguishable to ``.get(..., 0.0)`` — only treating a missing attr like a missing
    group tells them apart.
    """
    ep = _episode(tmp_path, with_time_sync_group=True, offset_s=None)
    with pytest.raises(RuntimeError, match='time_sync'):
        gopro_ts_in_finger_clock(ep, require_offset=True)


def test_missing_offset_tolerated_when_not_required(tmp_path: pathlib.Path) -> None:
    """A caller that resamples a slowly-varying signal accepts the unshifted grid instead."""
    ep = _episode(tmp_path, with_time_sync_group=True, offset_s=None)
    ts = gopro_ts_in_finger_clock(ep, require_offset=False)
    assert np.array_equal(ts, GOPRO_TS)


def test_missing_time_sync_group_tolerated_when_not_required(tmp_path: pathlib.Path) -> None:
    """Same tolerance applies when the group itself is entirely absent."""
    ep = _episode(tmp_path, with_time_sync_group=False, offset_s=None)
    ts = gopro_ts_in_finger_clock(ep, require_offset=False)
    assert np.array_equal(ts, GOPRO_TS)
