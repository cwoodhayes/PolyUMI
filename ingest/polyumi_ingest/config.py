"""Centralised configuration paths and loaders for polyumi_ingest."""

import pathlib

import yaml

# Root of the ingest/ directory.
INGEST_ROOT = pathlib.Path(__file__).parent.parent

GRIPPER_CALIB_YAML = INGEST_ROOT / 'config' / 'gripper_calib.yaml'


def load_gripper_calib() -> dict:
    """Load gripper calibration transforms from config/gripper_calib.yaml."""
    with GRIPPER_CALIB_YAML.open() as f:
        return yaml.safe_load(f)
