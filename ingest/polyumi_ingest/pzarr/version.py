"""pzarr version number."""

PZARR_VERSION = 2
"""
pzarr schema version.

v1: original schema (finger/gopro frames+audio, no IMU/GPS/optitrack arrays).
v2: adds gopro/{accl,gyro,gps} + timestamps/gopro_{accl,gyro,gps}, and the
    scene-level optitrack/{pose,timestamps} group.
"""
