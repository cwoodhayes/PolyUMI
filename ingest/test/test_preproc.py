"""Tests for preprocessing steps."""

import pathlib

import numpy as np
import zarr
from polyumi_pi import sync_chirp
from polyumi_ingest.preproc import ChirpTimeSyncStep, TimeSyncStep


def _sine_wave(freq_hz: float, sample_rate: int, duration_s: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32)


def test_time_sync_step_writes_offset_and_copy(tmp_path: pathlib.Path) -> None:
    """Verify TimeSyncStep writes the offset annotation and marks the step complete on copy."""
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ts = np.arange(0.0, 1.0, 1.0 / 16_000.0, dtype=np.float64)
    offset_s = 0.0125

    finger_audio = _sine_wave(440.0, 16_000, 1.0)
    gopro_audio = _sine_wave(440.0, 16_000, 1.0, phase=2.0 * np.pi * 440.0 * offset_s)

    ep.create_group('finger').create_array('finger_piezo', data=finger_audio)
    ep.create_group('gopro').create_array('audio', data=gopro_audio)
    ts_grp = ep.create_group('timestamps')
    ts_grp.create_array('finger_piezo', data=ts)
    ts_grp.create_array('gopro_audio', data=ts + offset_s)

    step = TimeSyncStep()
    output = step.run(scene_zarr, copy=True)

    assert output.name == 'scene_pp99.zarr'
    copied_root = zarr.open_group(str(output), mode='r')
    assert copied_root.attrs['preprocessing_steps'] == [99]

    ts_attrs = copied_root['episode_0/annotations/time_sync'].attrs  # type: ignore[index]
    offset = float(ts_attrs['gopro_to_finger_offset_s'])  # type: ignore[arg-type]
    residual = float(ts_attrs['residual_offset_s'])  # type: ignore[arg-type]
    lag_samples = int(ts_attrs['lag_samples'])  # type: ignore[arg-type]

    assert abs(offset - offset_s) < 0.02
    assert abs(residual) < 0.02
    assert abs(lag_samples) < 400


def test_chirp_time_sync_step(tmp_path: pathlib.Path) -> None:
    """ChirpTimeSyncStep recovers the inter-device offset from injected chirps."""
    sr = 16_000
    duration_s = 10.0
    true_offset_s = 0.75  # GoPro clock is 0.75 s ahead of finger clock
    start_delay_s = 2.0  # GoPro starts recording 2 s after the Pi (physical time)

    # Pi and GoPro have independent start times; GoPro clock = Pi clock + true_offset_s
    finger_start_s = 1_000.0
    gopro_start_s = finger_start_s + start_delay_s + true_offset_s

    chirp = sync_chirp.generate(sr)
    chirp_play_time_s = finger_start_s + 4.0  # chirp fires 4 s into finger recording

    rng = np.random.default_rng(0)
    n = int(duration_s * sr)

    finger_audio = rng.normal(0, 0.01, n).astype(np.float32)
    onset_finger = int((chirp_play_time_s - finger_start_s) * sr)
    finger_audio[onset_finger : onset_finger + len(chirp)] += chirp

    # In GoPro-clock, the chirp event happens at chirp_play_time_s + true_offset_s
    gopro_audio = rng.normal(0, 0.01, n).astype(np.float32)
    onset_gopro = int((chirp_play_time_s + true_offset_s - gopro_start_s) * sr)
    gopro_audio[onset_gopro : onset_gopro + len(chirp)] += chirp

    finger_ts = finger_start_s + np.arange(n, dtype=np.float64) / sr
    gopro_ts = gopro_start_s + np.arange(n, dtype=np.float64) / sr

    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ep.create_group('finger').create_array('finger_air', data=finger_audio)
    ep.create_group('gopro').create_array('audio', data=gopro_audio)
    ts_grp = ep.create_group('timestamps')
    ts_grp.create_array('finger_air', data=finger_ts)
    ts_grp.create_array('gopro_audio', data=gopro_ts)
    ep.create_group('annotations').attrs['sync_chirp_play_time_s'] = chirp_play_time_s

    ChirpTimeSyncStep().run(scene_zarr)

    root = zarr.open_group(str(scene_zarr), mode='r')
    offset = float(root['episode_0/annotations/time_sync'].attrs['gopro_to_finger_offset_s'])  # type: ignore[arg-type]
    assert abs(offset - true_offset_s) < 1.0 / sr  # sub-sample accuracy
