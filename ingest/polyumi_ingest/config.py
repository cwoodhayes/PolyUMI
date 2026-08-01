"""Centralised configuration paths and loaders for polyumi_ingest."""

import pathlib

import yaml

# Root of the ingest/ directory.
INGEST_ROOT = pathlib.Path(__file__).parent.parent

GRIPPER_CALIB_YAML = INGEST_ROOT / 'config' / 'gripper_calib.yaml'
GOPRO_INTRINSICS_JSON = INGEST_ROOT / 'config' / 'gopro_intrinsics.json'
#: Thresholds for auto-flagging episodes unusable. Policy, not data — verdicts are
#: derived at read time (see polyumi_ingest.quality), never written into the pzarr.
QUALITY_THRESHOLDS_YAML = INGEST_ROOT / 'config' / 'quality_thresholds.yaml'
#: How much data the SLAM step is fed (resolution divisor, localization frame stride)
#: and the reverse-merge gate. The camera model itself lives in the ORB-SLAM3 settings
#: YAML, which that file points at.
SLAM_CONFIG_YAML = INGEST_ROOT / 'config' / 'slam.yaml'


def load_gripper_calib() -> dict:
    """Load gripper calibration transforms from config/gripper_calib.yaml."""
    with GRIPPER_CALIB_YAML.open() as f:
        return yaml.safe_load(f)


def load_aruco_finger_config() -> dict:
    """Load the aruco_finger_tags section from gripper_calib.yaml."""
    return load_gripper_calib()['aruco_finger_tags']


def load_slam_config() -> dict:
    """
    Load SLAM step tunables from config/slam.yaml.

    Returns an empty dict if the file is missing so the step falls back to its
    in-code defaults rather than failing to import.
    """
    try:
        with SLAM_CONFIG_YAML.open() as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return {}
