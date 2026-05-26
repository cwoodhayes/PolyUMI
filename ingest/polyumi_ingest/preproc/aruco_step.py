"""ArUco-based 1DOF gripper-width preprocessing step."""

from __future__ import annotations

import json
import logging
import pathlib

import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from polyumi_ingest.config import GOPRO_INTRINSICS_JSON, load_aruco_finger_config
from polyumi_ingest.preproc._umi_cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    get_gripper_width,
    get_interp1d,
    parse_fisheye_intrinsics,
)
from polyumi_ingest.preproc.step_base import PreprocessingStep, register_preprocessing_step
from polyumi_ingest.pzarr.store import arr

log = logging.getLogger(__name__)

_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

# Below this fraction of frames with a valid detection, log a warning.
_MIN_DETECTION_RATE_WARN = 0.5


@register_preprocessing_step(step_number=4, step_name='aruco-gripper-width')
class ArucoGripperWidthStep(PreprocessingStep):
    """
    Estimate 1DOF gripper opening width from ArUco finger markers in GoPro frames.

    For each episode, runs ArUco detection on every GoPro frame, computes a 6DOF
    pose for the left and right finger tags via fisheye undistortion + solvePnP,
    and derives the gripper opening width from their x-coordinates in camera
    space.  Missing detections are linearly interpolated across the timestamp
    grid so the output array has one width per GoPro frame.

    Reuses the UMI implementation verbatim (see ``_umi_cv_util.py``).
    """

    def run_step(self, scene_zarr: pathlib.Path, force: bool = False) -> None:
        """Detect ArUco finger markers in each episode and write the gripper width series."""
        root = zarr.open_group(str(scene_zarr), mode='a')
        episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
        if not episodes:
            raise RuntimeError(f'No episodes found in {scene_zarr}')

        cfg = load_aruco_finger_config()
        left_id = int(cfg['left_id'])
        right_id = int(cfg['right_id'])
        marker_size_m = float(cfg['marker_size_m'])
        nominal_z_m = float(cfg['nominal_z_m'])
        z_tolerance_m = float(cfg['z_tolerance_m'])

        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg['dictionary']))
        marker_size_map = {left_id: marker_size_m, right_id: marker_size_m}

        with GOPRO_INTRINSICS_JSON.open() as f:
            base_intr = parse_fisheye_intrinsics(json.load(f))

        for episode_key in episodes:
            ep = root.require_group(episode_key)
            self._process_episode(
                ep=ep,
                episode_key=episode_key,
                base_intr=base_intr,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                left_id=left_id,
                right_id=right_id,
                marker_size_m=marker_size_m,
                nominal_z_m=nominal_z_m,
                z_tolerance_m=z_tolerance_m,
                force=force,
            )

    def _process_episode(
        self,
        ep: zarr.Group,
        episode_key: str,
        base_intr: dict,
        aruco_dict: cv2.aruco.Dictionary,
        marker_size_map: dict[int, float],
        left_id: int,
        right_id: int,
        marker_size_m: float,
        nominal_z_m: float,
        z_tolerance_m: float,
        force: bool,
    ) -> None:
        if 'gopro/frames' not in ep:
            log.warning(f'{episode_key}: no gopro/frames; skipping aruco width.')
            return

        frames_arr = arr(ep, 'gopro/frames')
        timestamps = np.asarray(arr(ep, 'timestamps/gopro')[:], dtype=np.float64)
        n_frames, H, W, _ = frames_arr.shape
        if n_frames != len(timestamps):
            raise RuntimeError(f'{episode_key}: frame count {n_frames} != timestamp count {len(timestamps)}')

        # Scale calibration to the actual frame resolution.
        intr = convert_fisheye_intrinsics_resolution(base_intr, (W, H))

        raw_ts: list[float] = []
        raw_widths: list[float] = []
        for i in tqdm(range(n_frames), desc=f'{episode_key} aruco', unit='frame'):
            frame = np.asarray(frames_arr[i])
            tag_dict = detect_localize_aruco_tags(
                frame,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=intr,
            )
            width = get_gripper_width(
                tag_dict,
                left_id=left_id,
                right_id=right_id,
                nominal_z=nominal_z_m,
                z_tolerance=z_tolerance_m,
            )
            if width is not None:
                raw_ts.append(float(timestamps[i]))
                raw_widths.append(float(width))

        n_detected = len(raw_widths)
        detection_rate = n_detected / n_frames if n_frames > 0 else 0.0
        if detection_rate < _MIN_DETECTION_RATE_WARN:
            log.warning(
                f'{episode_key}: only {detection_rate:.1%} of frames had a valid '
                f'finger-tag detection ({n_detected}/{n_frames}); width series may be unreliable.'
            )

        if n_detected >= 2:
            interp = get_interp1d(np.array(raw_ts), np.array(raw_widths))
            width_m = interp(timestamps).astype(np.float32)
        elif n_detected == 1:
            width_m = np.full(n_frames, raw_widths[0], dtype=np.float32)
        else:
            width_m = np.full(n_frames, np.nan, dtype=np.float32)

        out_grp = ep.require_group('annotations').require_group('gripper_width')
        for arr_name in ('width_m', 'raw_widths_m', 'raw_timestamps_s'):
            if arr_name in out_grp:
                del out_grp[arr_name]
        out_grp.create_array('width_m', data=width_m, compressor=_BLOSC)
        out_grp.create_array(
            'raw_widths_m',
            data=np.array(raw_widths, dtype=np.float32),
            compressor=_BLOSC,
        )
        out_grp.create_array(
            'raw_timestamps_s',
            data=np.array(raw_ts, dtype=np.float64),
            compressor=_BLOSC,
        )

        out_grp.attrs['detection_rate'] = float(detection_rate)
        out_grp.attrs['n_detected'] = int(n_detected)
        out_grp.attrs['n_frames'] = int(n_frames)
        out_grp.attrs['left_id'] = left_id
        out_grp.attrs['right_id'] = right_id
        out_grp.attrs['marker_size_m'] = marker_size_m
        out_grp.attrs['nominal_z_m'] = nominal_z_m
        out_grp.attrs['z_tolerance_m'] = z_tolerance_m

        log.info(
            f'{episode_key}: aruco width — '
            f'{n_detected}/{n_frames} detections ({detection_rate:.1%})'
            + (
                f', width range [{np.nanmin(width_m) * 1000:.1f}, {np.nanmax(width_m) * 1000:.1f}] mm'
                if n_detected > 0
                else ''
            )
        )
