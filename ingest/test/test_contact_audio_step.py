"""Tests for the contact-mic preprocessing step."""

import pathlib

import numpy as np
import pytest
import zarr
from polyumi_ingest.config import load_contact_audio_config
from polyumi_ingest.preproc import ContactAudioStep

SR = 16_000
FPS = 59.94
#: Piezo epoch. Arbitrary but non-zero, so a test can't pass by treating the finger clock as
#: starting at 0 — which is exactly the bug the clock hop exists to prevent.
PIEZO_START_S = 1_000.0
BLOCK_WIDTH = int(load_contact_audio_config()['blocks']['samples_per_gopro_frame'])


def _build_scene(
    tmp_path: pathlib.Path,
    *,
    n_frames: int = 120,
    offset_s: float = 0.5,
    lead_s: float = 0.25,
    with_time_sync: bool = True,
    with_piezo: bool = True,
    session_type: str = 'EPISODE',
    drop_frame: int | None = None,
    duration_pad_s: float = 1.0,
) -> pathlib.Path:
    """
    Build a minimal scene.zarr whose piezo sample *i* has value *i*.

    That identity is what lets the assertions below talk about which source samples landed in
    which block without threading an expected array through every test.

    ``offset_s`` (the chirp offset the GoPro clock leads the finger clock by) and ``lead_s``
    (how far into the recording the first GoPro frame falls) are deliberately different
    numbers, so an anchor computed without the clock hop, or with its sign flipped, lands
    somewhere the assertions can see.
    """
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ep.attrs['session_dir'] = 'session_0'
    ep.attrs['session_type'] = session_type

    gopro_ts = PIEZO_START_S + lead_s + offset_s + np.arange(n_frames, dtype=np.float64) / FPS
    if drop_frame is not None:
        gopro_ts = np.delete(gopro_ts, drop_frame)
    ts_grp = ep.create_group('timestamps')
    ts_grp.create_array('gopro', data=gopro_ts)

    if with_piezo:
        n_samples = int((lead_s + n_frames / FPS + duration_pad_s) * SR)
        piezo = np.arange(n_samples, dtype=np.float32)
        piezo_ts = PIEZO_START_S + np.arange(n_samples, dtype=np.float64) / SR
        ep.create_group('finger').create_array('finger_piezo', data=piezo)
        ts_grp.create_array('finger_piezo', data=piezo_ts)

    if with_time_sync:
        ep.require_group('annotations').require_group('time_sync').attrs['gopro_to_finger_offset_s'] = offset_s
    return scene_zarr


def _out(scene_zarr: pathlib.Path) -> zarr.Group:
    return zarr.open_group(str(scene_zarr), mode='r')['episode_0/annotations/contact_audio']  # type: ignore[index,return-value]


def test_blocks_are_anchored_by_timestamp(tmp_path: pathlib.Path) -> None:
    """Each block starts at the piezo sample matching its GoPro frame, in the finger clock."""
    scene_zarr = _build_scene(tmp_path)
    ContactAudioStep().run(scene_zarr)

    out = _out(scene_zarr)
    starts = np.asarray(out['frame_block_start_idx'][:])  # type: ignore[index]
    blocks = np.asarray(out['frame_blocks'][:])  # type: ignore[index]

    assert blocks.shape == (len(starts), BLOCK_WIDTH)
    assert blocks.dtype == np.float32
    # piezo[i] == i, so a block's first value is the source index it was anchored at. This is the
    # assertion that fails if the chirp offset is dropped or applied with the wrong sign.
    assert np.array_equal(blocks[:, 0], starts.astype(np.float32))
    # Frame 0 sits lead_s into the piezo recording once the chirp offset is removed. Dropping
    # the offset would put it at (lead_s + offset_s) * SR, and flipping its sign further still.
    assert starts[0] == pytest.approx(0.25 * SR, abs=1)
    # Non-integer spacing: 16000/59.94 = 266.93 means adjacent, not equal, integer gaps.
    assert set(np.unique(np.diff(starts))) <= {266, 267}
    assert zarr.open_group(str(scene_zarr), mode='r').attrs['preprocessing_steps'] == [6]


def test_consecutive_blocks_never_gap(tmp_path: pathlib.Path) -> None:
    """The invariant the exporter's contiguity rests on: block k reaches block k+1's start."""
    scene_zarr = _build_scene(tmp_path, n_frames=600)
    ContactAudioStep().run(scene_zarr)

    starts = np.asarray(_out(scene_zarr)['frame_block_start_idx'][:])  # type: ignore[index]
    assert (starts[:-1] + BLOCK_WIDTH >= starts[1:]).all()


def test_flattened_blocks_cover_every_sample(tmp_path: pathlib.Path) -> None:
    """Concatenating consecutive blocks yields a waveform with no source sample missing."""
    scene_zarr = _build_scene(tmp_path, n_frames=300)
    ContactAudioStep().run(scene_zarr)

    out = _out(scene_zarr)
    blocks = np.asarray(out['frame_blocks'][:])  # type: ignore[index]
    starts = np.asarray(out['frame_block_start_idx'][:])  # type: ignore[index]

    seen = np.unique(blocks.reshape(-1).astype(np.int64))
    expected = np.arange(starts[0], starts[-1] + BLOCK_WIDTH)
    assert np.array_equal(np.intersect1d(seen, expected), expected)


def test_dropped_frame_is_counted_not_repaired(tmp_path: pathlib.Path) -> None:
    """A missing GoPro frame leaves a real hole; the step reports it rather than inventing audio."""
    scene_zarr = _build_scene(tmp_path, drop_frame=50)
    ContactAudioStep().run(scene_zarr)

    out = _out(scene_zarr)
    assert int(out.attrs['n_frame_gaps']) == 1  # type: ignore[arg-type]
    assert int(out.attrs['max_frame_spacing_samples']) > BLOCK_WIDTH  # type: ignore[arg-type]


def test_tail_is_zero_filled_and_counted(tmp_path: pathlib.Path) -> None:
    """Blocks reading past the end of the recording pad with zeros and say how much."""
    # duration_pad_s=0 leaves the last frames' blocks hanging off the end of the piezo array.
    scene_zarr = _build_scene(tmp_path, duration_pad_s=0.0)
    ContactAudioStep().run(scene_zarr)

    out = _out(scene_zarr)
    blocks = np.asarray(out['frame_blocks'][:])  # type: ignore[index]
    assert int(out.attrs['n_zero_filled_samples']) > 0  # type: ignore[arg-type]
    assert blocks[-1, -1] == 0.0


def test_missing_time_sync_flags_the_episode(tmp_path: pathlib.Path) -> None:
    """Without step 1 the two clocks are unrelated epochs, so the step must refuse."""
    scene_zarr = _build_scene(tmp_path, with_time_sync=False)
    ContactAudioStep().run(scene_zarr)

    ep = zarr.open_group(str(scene_zarr), mode='r')['episode_0']
    assert 'contact_audio' not in ep.get('annotations', {})  # type: ignore[union-attr]
    failure = ep.attrs['failure']  # type: ignore[index]
    assert failure['step'] == 'contact-audio'
    assert 'time_sync' in failure['error']


def test_missing_piezo_is_skipped_not_flagged(tmp_path: pathlib.Path) -> None:
    """An episode recorded without the contact mic isn't broken; it just has no audio."""
    scene_zarr = _build_scene(tmp_path, with_piezo=False)
    ContactAudioStep().run(scene_zarr)

    ep = zarr.open_group(str(scene_zarr), mode='r')['episode_0']
    assert 'failure' not in ep.attrs
    assert 'annotations' not in ep or 'contact_audio' not in ep['annotations']  # type: ignore[operator]


def test_mapping_session_is_skipped(tmp_path: pathlib.Path) -> None:
    """The mapping session carries no demonstration, so it needs no audio blocks."""
    scene_zarr = _build_scene(tmp_path, session_type='MAPPING')
    ContactAudioStep().run(scene_zarr)

    ep = zarr.open_group(str(scene_zarr), mode='r')['episode_0']
    assert 'failure' not in ep.attrs
    assert 'annotations' not in ep or 'contact_audio' not in ep['annotations']  # type: ignore[operator]


def test_sample_rate_mismatch_is_fatal(tmp_path: pathlib.Path) -> None:
    """Timestamps implying a different rate would give every step the wrong duration of audio."""
    scene_zarr = _build_scene(tmp_path)
    root = zarr.open_group(str(scene_zarr), mode='a')
    n = root['episode_0/timestamps/finger_piezo'].shape[0]  # type: ignore[index,union-attr]
    del root['episode_0/timestamps/finger_piezo']
    root['episode_0/timestamps'].create_array(  # type: ignore[union-attr]
        'finger_piezo', data=PIEZO_START_S + np.arange(n, dtype=np.float64) / 48_000.0
    )

    ContactAudioStep().run(scene_zarr)

    failure = zarr.open_group(str(scene_zarr), mode='r')['episode_0'].attrs['failure']  # type: ignore[index]
    assert '48000' in failure['error'] or '48000.0' in failure['error']


def test_backward_timestamps_are_fatal_even_if_the_rate_would_infer_correctly(tmp_path: pathlib.Path) -> None:
    """
    A duplicate or backward sample must not slip through just because most deltas look right.

    `_gather_blocks` anchors every block with `searchsorted`, which assumes `piezo_ts` is fully
    sorted — a single non-positive delta buried among otherwise-correct ones would silently mis-
    anchor the block(s) around it rather than fail loudly.
    """
    scene_zarr = _build_scene(tmp_path)
    root = zarr.open_group(str(scene_zarr), mode='a')
    piezo_ts = np.asarray(root['episode_0/timestamps/finger_piezo'][:])  # type: ignore[index,union-attr]
    piezo_ts[len(piezo_ts) // 2] = piezo_ts[len(piezo_ts) // 2 - 1]  # one duplicated sample
    del root['episode_0/timestamps/finger_piezo']
    root['episode_0/timestamps'].create_array('finger_piezo', data=piezo_ts)  # type: ignore[union-attr]

    ContactAudioStep().run(scene_zarr)

    failure = zarr.open_group(str(scene_zarr), mode='r')['episode_0'].attrs['failure']  # type: ignore[index]
    assert 'strictly increasing' in failure['error']


def test_diagnostic_logmel_is_written(tmp_path: pathlib.Path) -> None:
    """The mel is a by-product for the catalog; it still has to be well-formed."""
    scene_zarr = _build_scene(tmp_path)
    ContactAudioStep().run(scene_zarr)

    out = _out(scene_zarr)
    cfg = load_contact_audio_config()['logmel']
    logmel = np.asarray(out['logmel'][:])  # type: ignore[index]
    logmel_ts = np.asarray(out['logmel_timestamps'][:])  # type: ignore[index]

    assert logmel.shape[1] == cfg['n_mels']
    assert logmel.shape[0] == len(logmel_ts)
    assert logmel.dtype == np.float32
    assert np.isfinite(logmel).all()
    assert (np.diff(logmel_ts) > 0).all()
    assert int(out.attrs['logmel_n_mels']) == cfg['n_mels']  # type: ignore[arg-type]
