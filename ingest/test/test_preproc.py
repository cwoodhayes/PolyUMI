import pathlib

import numpy as np
import zarr
from polyumi_ingest.preproc import TimeSyncStep


def _sine_wave(freq_hz: float, sample_rate: int, duration_s: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32)


def test_time_sync_step_writes_offset_and_copy(tmp_path: pathlib.Path) -> None:
    scene_zarr = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    ep = root.create_group('episode_0')
    ts = np.arange(0.0, 1.0, 1.0 / 16_000.0, dtype=np.float64)
    offset_s = 0.0125

    finger_audio = _sine_wave(440.0, 16_000, 1.0)
    gopro_audio = _sine_wave(440.0, 16_000, 1.0, phase=2.0 * np.pi * 440.0 * offset_s)

    ep.create_group('finger').create_array('finger_air', data=finger_audio)
    ep.create_group('gopro').create_array('audio', data=gopro_audio)
    ts_grp = ep.create_group('timestamps')
    ts_grp.create_array('finger_air', data=ts)
    ts_grp.create_array('gopro_audio', data=ts + offset_s)

    step = TimeSyncStep()
    output = step.run(scene_zarr, copy=True)

    assert output.name == 'scene_pp1.zarr'
    copied_root = zarr.open_group(str(output), mode='r')
    assert copied_root.attrs['preprocessing_steps'] == [1]

    offset = float(copied_root['episode_0/annotations/time_sync/gopro_audio_to_finger_air_offset_s'][()])
    residual = float(copied_root['episode_0/annotations/time_sync/residual_offset_s'][()])
    lag_samples = int(copied_root['episode_0/annotations/time_sync/lag_samples'][()])

    assert abs(offset - offset_s) < 0.02
    assert abs(residual) < 0.02
    assert abs(lag_samples) < 400
