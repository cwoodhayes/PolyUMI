"""Tests for the /finger/logmel diagnostic spectrogram channel exported to MCAP for Foxglove."""

import base64
import json
import pathlib

import cv2
import numpy as np
import pytest
import zarr

from polyumi_ingest.export.mcap import _LOGMEL_PUBLISH_HZ, _LOGMEL_WINDOW_S, export_episode_to_mcap

HOP_S = 0.01
N_MELS = 64


def _build_episode(tmp_path: pathlib.Path, *, n_hops: int | None) -> tuple[zarr.Group, zarr.Group]:
    """Build a minimal episode, optionally carrying a step-6 contact_audio annotation."""
    n = 2
    root = zarr.open_group(str(tmp_path / 'scene.zarr'), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')

    ep.create_group('finger').create_array('frames', data=np.zeros((n, 8, 8, 3), dtype=np.uint8))
    ep['finger'].create_array('finger_piezo', data=np.zeros(64, dtype=np.float32))  # type: ignore[union-attr]
    ep['finger'].create_array('finger_air', data=np.zeros(64, dtype=np.float32))  # type: ignore[union-attr]
    ts_grp = ep.create_group('timestamps')
    ts_grp.create_array('finger', data=np.arange(n, dtype=np.float64) / 10.0)
    ts_grp.create_array('finger_piezo', data=np.arange(64, dtype=np.float64) / 16_000.0)
    ts_grp.create_array('finger_air', data=np.arange(64, dtype=np.float64) / 16_000.0)

    if n_hops is not None:
        grp = ep.create_group('annotations').create_group('contact_audio')
        # A loud band in the top mel bins halfway through, so orientation is testable.
        logmel = np.full((n_hops, N_MELS), -13.8, dtype=np.float32)
        if n_hops:
            logmel[n_hops // 2, N_MELS - 4 :] = 3.0
        grp.create_array('logmel', data=logmel)
        grp.create_array('logmel_timestamps', data=np.arange(n_hops, dtype=np.float64) * HOP_S)

    return root, ep


def _logmel_msgs(mcap_path: pathlib.Path) -> list[dict]:
    from mcap.reader import make_reader

    with mcap_path.open('rb') as f:
        return [json.loads(msg.data) for _, _, msg in make_reader(f).iter_messages(topics=['/finger/logmel'])]


def _decode(msg: dict) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(msg['data']), dtype=np.uint8), cv2.IMREAD_COLOR)


def test_publishes_at_the_configured_rate_on_the_hop_clock(tmp_path: pathlib.Path) -> None:
    """One message per publish tick, stamped with the hop it is centred on."""
    n_hops = 500  # 5 s at a 10 ms hop
    root, ep = _build_episode(tmp_path, n_hops=n_hops)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    msgs = _logmel_msgs(mcap_path)
    stride = round(1.0 / (_LOGMEL_PUBLISH_HZ * HOP_S))
    assert len(msgs) == -(-n_hops // stride)
    first = msgs[0]['timestamp']['sec'] + msgs[0]['timestamp']['nsec'] / 1e9
    second = msgs[1]['timestamp']['sec'] + msgs[1]['timestamp']['nsec'] / 1e9
    assert first == pytest.approx(0.0, abs=1e-6)
    assert second - first == pytest.approx(stride * HOP_S, abs=1e-6)


def test_window_geometry_is_constant_including_at_the_edges(tmp_path: pathlib.Path) -> None:
    """
    Every window is the same size, so the playhead column never drifts.

    The first and last ticks have less than half a window of real data on one side; padding
    rather than clipping is what keeps "now" at the centre column throughout.
    """
    root, ep = _build_episode(tmp_path, n_hops=500)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    shapes = {_decode(m).shape for m in _logmel_msgs(mcap_path)}
    assert len(shapes) == 1
    height, width, _ = shapes.pop()
    assert width == round(_LOGMEL_WINDOW_S / HOP_S) * 2  # _LOGMEL_ZOOM_X
    assert height == N_MELS * 4  # _LOGMEL_ZOOM_Y


def test_low_frequencies_are_drawn_at_the_bottom(tmp_path: pathlib.Path) -> None:
    """The loud top-of-band burst must land in the upper rows, not the lower ones."""
    n_hops = 500
    root, ep = _build_episode(tmp_path, n_hops=n_hops)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    # The tick whose window is centred on the loud hop: brightest image of the run.
    brightest = max(_logmel_msgs(mcap_path), key=lambda m: _decode(m).mean())
    img = _decode(brightest)
    top_half, bottom_half = img[: img.shape[0] // 2], img[img.shape[0] // 2 :]
    assert top_half.mean() > bottom_half.mean()


def test_no_channel_without_the_annotation(tmp_path: pathlib.Path) -> None:
    """Step 6 never ran — no /finger/logmel messages, and the export still succeeds."""
    root, ep = _build_episode(tmp_path, n_hops=None)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    assert _logmel_msgs(mcap_path) == []


def test_no_channel_for_an_empty_spectrogram(tmp_path: pathlib.Path) -> None:
    """An episode too short for one FFT frame leaves a (0, n_mels) array; that is not a channel."""
    root, ep = _build_episode(tmp_path, n_hops=0)
    mcap_path = tmp_path / 'episode.mcap'

    export_episode_to_mcap(ep, mcap_path, root_grp=root)

    assert _logmel_msgs(mcap_path) == []
