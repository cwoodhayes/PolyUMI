"""End-effector pose preprocessing step: puts both pose sources on the canonical hand frame."""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import zarr
from numcodecs import Blosc

from polyumi_ingest.config import load_gripper_calib
from polyumi_ingest.interpolation import interpolate_se3_gaps
from polyumi_ingest.preproc.step_base import PreprocessingStep, register_preprocessing_step
from polyumi_ingest.pzarr.store import arr, grp
from polyumi_ingest.timebase import nearest_idx
from polyumi_ingest.transforms import (
    gopro_to_hand_transform,
    gripper_calib_transforms,
    retarget_body_frame,
)

log = logging.getLogger(__name__)

_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

#: Preference order when a scene carries more than one pose source. OptiTrack is mocap ground
#: truth and beats SLAM wherever the volume covers the demo.
_SOURCE_PREFERENCE = ('optitrack', 'slam')

#: Longest interior SLAM tracking-loss gap (in GoPro frames) that gets Slerp+linear filled.
#: A synthetic hold-out study on real trajectories put median error at a 20-frame gap around
#: ~8 mm / ~1.2°, degrading fast beyond that; larger gaps are left NaN and trimmed downstream.
_SLAM_MAX_GAP_FRAMES = 20


def _gopro_ts_in_finger_clock(ep: zarr.Group) -> np.ndarray:
    """GoPro frame timestamps shifted into the finger (= OptiTrack) clock domain."""
    ts = np.asarray(arr(ep, 'timestamps/gopro')[:], dtype=np.float64)
    if 'annotations/time_sync' in ep:
        offset = float(grp(ep, 'annotations/time_sync').attrs.get('gopro_to_finger_offset_s', 0.0))
        ts = ts - offset
    return ts


@register_preprocessing_step(step_number=5, step_name='eef-pose')
class EefPoseStep(PreprocessingStep):
    """
    Re-express each episode's pose trajectory onto the canonical hand frame.

    Neither raw pose source is in a frame the policy can use. OptiTrack reports the pose of the
    marker *rigid body*, whose origin Motive places at the marker centroid — an arbitrary point
    that moves whenever the markers are re-stuck. SLAM reports the GoPro optical frame. Neither
    coincides with the frame the robot reports at inference, and the two are not even in the
    same frame as each other, so models trained from different sources are not comparable.

    Both sources are converted onto one canonical **body** frame — the hand — and written to
    **one array per available source**, ``<episode>/eef/pose_optitrack`` and/or
    ``<episode>/eef/pose_slam``, on the GoPro frame grid (the same grid as ``gopro/frames`` and
    ``annotations/gripper_width``, so a single index serves all three downstream). Writing both
    alternates (rather than baking in a single winner) lets pose-source selection move to
    **export time** — see ``export.dp.buffer`` — instead of being frozen at preprocessing time.
    ``eef.attrs['default_source']`` records which one export uses absent an override.

    The chain routes both sources through the **GoPro frame** and then applies one shared
    ``T_gopro_to_hand`` hop::

        slam:      T_s_gp  ──────────────────────────────────► · T_gp_hand
        optitrack: T_o_rb · inv(T_gb_rb) · T_gb_gp  ─────────► · T_gp_hand

    The GoPro is the pivot because it is the only thing both embodiments share. ``gripper_base``
    is a mechanical part of the *handheld* gripper that the Franka end-effector does not have,
    so a frame defined against it could never be reconstructed on the robot; it survives here
    only as an intermediate in the OptiTrack chain, where it is valid because the markers really
    are mounted on that part. The GoPro-to-fingers geometry is identical across both, so a hand
    frame defined against the GoPro is reproducible on either.

    The *world* frame is deliberately left alone: OptiTrack-sourced poses stay in the OptiTrack
    frame and SLAM-sourced poses in the SLAM frame. A shared world frame cancels out of the
    relative pose representation the policy trains on, so normalizing it would be busywork;
    the body frame does not cancel, which is why it has to be fixed here. See
    ``transforms.retarget_body_frame``.

    SLAM leaves NaN rows wherever tracking was lost. Short *interior* gaps (a valid pose on
    both sides, up to ``_SLAM_MAX_GAP_FRAMES``) are filled with Slerp+linear interpolation via
    ``interpolation.interpolate_se3_gaps`` before retargeting, so a brief dropout does not
    truncate the usable span; leading/trailing NaN and longer gaps stay NaN by design and are
    trimmed downstream. OptiTrack is dense and needs no such filling.

    Runs after step 3 (slam-optitrack-align), which needs the untouched source-frame poses to
    solve for T_ws, and step 4 (aruco-gripper-width), which defines the GoPro-grid convention.

    Prerequisites: ``T_gopro_to_hand`` in ``config/gripper_calib.yaml``; ``timestamps/gopro``
    per episode; and either ``optitrack/pose`` + ``optitrack/timestamps`` in the root group or
    ``gopro/slam_poses`` in the episode.
    """

    def run_step(self, scene_zarr: pathlib.Path, force: bool = False) -> None:
        """Write ``eef/pose_<source>`` for every source each episode can supply."""
        root = zarr.open_group(str(scene_zarr), mode='a')
        episodes = sorted(k for k in root.keys() if k.startswith('episode_'))
        if not episodes:
            raise RuntimeError(f'No episodes found in {scene_zarr}')

        gripper_calib = load_gripper_calib()
        T_gb_rb, T_gb_gp, _ = gripper_calib_transforms(gripper_calib)
        T_gp_hand = gopro_to_hand_transform(gripper_calib)
        root.attrs['gripper_calib'] = gripper_calib

        # Per-source hop from what the sensor reports to the hand frame. Both route through the
        # GoPro frame, which is the only body both embodiments share; see the class docstring.
        source_to_hand = {
            'slam': T_gp_hand,
            'optitrack': T_gb_rb.inv() * T_gb_gp * T_gp_hand,
        }

        for episode_key in episodes:
            ep = root.require_group(episode_key)
            self._process_episode(root, ep, episode_key, source_to_hand, force=force)

    def _available_sources(self, root: zarr.Group, ep: zarr.Group) -> list[str]:
        """Pose sources this episode can actually supply, in preference order."""
        available = []
        if 'optitrack/pose' in root and 'optitrack/timestamps' in root:
            available.append('optitrack')
        if 'gopro/slam_poses' in ep:
            available.append('slam')
        return [s for s in _SOURCE_PREFERENCE if s in available]

    def _process_episode(
        self,
        root: zarr.Group,
        ep: zarr.Group,
        episode_key: str,
        source_to_hand: dict,
        force: bool,
    ) -> None:
        """Resolve this episode's available pose sources and write one eef/pose_<source> each."""
        sources = self._available_sources(root, ep)
        if not sources:
            log.warning(f'  {episode_key}: no optitrack or slam pose source; skipping.')
            return

        existing = set(ep['eef'].attrs.get('available_sources', [])) if 'eef' in ep else set()
        if not force and existing.issuperset(sources):
            log.info(f'  {episode_key}: eef/pose_* already present for {sources}; use --force to recompute.')
            return

        gopro_ts = _gopro_ts_in_finger_clock(ep)
        out_grp = ep.require_group('eef')

        # Clean up the pre-dual-source schema: a scene preprocessed by the old EefPoseStep (a
        # single eef/pose array plus group-level 'source'/'world_frame'/'n_nan' attrs) leaves
        # both behind on a --force re-run otherwise, since neither is part of the new schema
        # and nothing above would ever remove them.
        if 'pose' in out_grp:
            del out_grp['pose']
        for stale_attr in ('source', 'world_frame', 'n_nan'):
            out_grp.attrs.pop(stale_attr, None)

        for source in sources:
            n_interp = 0
            if source == 'optitrack':
                # OptiTrack runs on its own clock and rate; resample it onto the GoPro frame
                # grid so eef/pose_optitrack shares one index with the frames and gripper width.
                opti_ts = np.asarray(arr(root, 'optitrack/timestamps')[:], dtype=np.float64)
                opti_poses = np.asarray(arr(root, 'optitrack/pose')[:], dtype=np.float64)
                raw = opti_poses[nearest_idx(opti_ts, gopro_ts)]
                world_frame = 'optitrack'
            else:
                # SLAM leaves NaN rows wherever tracking was lost. Fill short interior gaps
                # (Slerp rotation + linear translation) so a brief dropout doesn't truncate the
                # usable span; long gaps and the leading/trailing NaN stay NaN by design.
                raw = np.asarray(arr(ep, 'gopro/slam_poses')[:], dtype=np.float64)
                raw, filled = interpolate_se3_gaps(raw, max_gap_frames=_SLAM_MAX_GAP_FRAMES)
                n_interp = int(filled.sum())
                world_frame = 'slam'

            if len(raw) != len(gopro_ts):
                raise RuntimeError(
                    f'{episode_key}: pose source {source!r} has {len(raw)} rows but the gopro '
                    f'grid has {len(gopro_ts)}; refusing to write a misaligned eef/pose_{source}.'
                )

            pose = retarget_body_frame(raw, source_to_hand[source])
            n_nan = int(np.isnan(pose[:, 0]).sum())

            array_name = f'pose_{source}'
            if array_name in out_grp:
                del out_grp[array_name]
            pose_arr = out_grp.create_array(array_name, data=pose, compressor=_BLOSC)
            pose_arr.attrs['world_frame'] = world_frame
            pose_arr.attrs['body_frame'] = 'hand'
            pose_arr.attrs['grid'] = 'gopro'
            pose_arr.attrs['n_nan'] = n_nan
            pose_arr.attrs['n_interp_filled'] = n_interp

            interp_note = f', {n_interp} interp-filled' if n_interp else ''
            log.info(
                f'  {episode_key}: eef/pose_{source} {pose.shape} '
                f'(world={world_frame}, body=hand, {n_nan}/{len(pose)} NaN{interp_note})'
            )

        out_grp.attrs['available_sources'] = sources
        out_grp.attrs['default_source'] = sources[0]  # _SOURCE_PREFERENCE order
        out_grp.attrs['body_frame'] = 'hand'
        out_grp.attrs['grid'] = 'gopro'
