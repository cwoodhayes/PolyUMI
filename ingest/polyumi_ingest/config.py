"""Centralised configuration paths and loaders for polyumi_ingest."""

import pathlib

import yaml

# Root of the ingest/ directory.
INGEST_ROOT = pathlib.Path(__file__).parent.parent

GRIPPER_CALIB_YAML = INGEST_ROOT / 'config' / 'gripper_calib.yaml'
GOPRO_INTRINSICS_JSON = INGEST_ROOT / 'config' / 'gopro_intrinsics.json'


def load_gripper_calib() -> dict:
    """Load gripper calibration transforms from config/gripper_calib.yaml."""
    with GRIPPER_CALIB_YAML.open() as f:
        return yaml.safe_load(f)


def load_aruco_finger_config() -> dict:
    """Load the aruco_finger_tags section from gripper_calib.yaml."""
    return load_gripper_calib()['aruco_finger_tags']
