"""pzarr version number."""

PZARR_VERSION = 3
"""
pzarr schema version.

v1: original schema (finger/gopro frames+audio, no IMU/GPS/optitrack arrays).
v2: adds gopro/{accl,gyro,gps} + timestamps/gopro_{accl,gyro,gps}, and the
    scene-level optitrack/{pose,timestamps} group.
v3: drops the gopro/frames array — GoPro frames are decoded on demand from the
    gopro.mp4 sidecar (video_helpers.GoproMp4Frames), which the pzarr no longer
    re-encodes as JpegXl (~70x smaller on disk). timestamps/gopro still defines
    the authoritative GoPro frame grid. finger/frames is unchanged.
"""
