"""pzarr version number."""

PZARR_VERSION = 4
"""
pzarr schema version.

v1: original schema (finger/gopro frames+audio, no IMU/GPS/optitrack arrays).
v2: adds gopro/{accl,gyro,gps} + timestamps/gopro_{accl,gyro,gps}, and the
    scene-level optitrack/{pose,timestamps} group.
v3: drops the gopro/frames array — GoPro frames are decoded on demand from the
    gopro.mp4 sidecar (video_helpers.GoproMp4Frames), which the pzarr no longer
    re-encodes as JpegXl (~70x smaller on disk). timestamps/gopro still defines
    the authoritative GoPro frame grid. finger/frames is unchanged.
v4: SLAM output is forward-only — gopro/slam_poses_{forward,reverse} and the
    annotations/slam reverse_* attrs are gone. annotations/slam gains the fed-grid
    post-chirp counts the usability gate reads. eef/pose_* is no longer gap-filled,
    so it carries NaN wherever SLAM had no pose (at frame_stride 2, every un-fed
    frame) and n_interp_filled is gone. gopro/slam_poses is also now in the GoPro
    **optical** frame (x right, y down, z forward) rather than the IMU body frame:
    the binaries switched from SaveTrajectoryEuRoC, whose inertial branch composes
    mTbc, to SaveTrajectoryCSV, which reports Twc — the frame upstream UMI trains
    against, and the one gripper_calib.yaml's transforms were always written for.

Unlike v1-v3, which changed what ``build_pzarr`` writes, v4 changes only the output
of preprocessing steps 2 and 5 — so migrating a store needs a full ``pingest pp
--force``, not a rebuild. ``run_preprocessing`` restamps the attr once every step has
actually run; see ``preproc.step_base``.
"""
