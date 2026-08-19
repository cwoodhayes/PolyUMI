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
#: How much data the SLAM step is fed: resolution divisor and localization frame stride.
#: The camera model itself lives in the ORB-SLAM3 settings YAML, which that file points at.
SLAM_CONFIG_YAML = INGEST_ROOT / 'config' / 'slam.yaml'


def load_gripper_calib() -> dict:
    """Load gripper calibration transforms from config/gripper_calib.yaml."""
    with GRIPPER_CALIB_YAML.open() as f:
        return yaml.safe_load(f)


def load_aruco_finger_config() -> dict:
    """Load the aruco_finger_tags section from gripper_calib.yaml."""
    return load_gripper_calib()['aruco_finger_tags']


def load_closed_width_m() -> float:
    """
    Load the closed width: the ArUco finger-tag separation with the gripper fully closed, in metres.

    The DP exporter subtracts this so exported widths are opening-from-closed rather than raw tag
    separation, matching UMI (whose ``get_gripper_calibration_interpolator``, in the upstream repo's
    ``umi/common/interpolation_util.py``, does the same
    subtraction at dataset-generation time). Stored in millimetres because that is the unit it is
    measured and reasoned about in.

    Required, not defaulted: a wrong-by-default value here shifts every exported width by that
    amount and there is nothing downstream that would notice. Derive it with
    ``pingest calibrate-gripper``.

    :raises KeyError: if ``gripper_fingers.closed_mm`` is missing from the config.
    """
    return float(load_gripper_calib()['gripper_fingers']['closed_mm']) / 1000.0


def load_open_width_m() -> float:
    """
    Load the open width: the ArUco finger-tag separation with the gripper fully open, in metres.

    Paired with :func:`load_closed_width_m` to bound exported widths. The handheld gripper is a
    rigid mechanism with a hard stop, so a tag separation above this is a detection failure
    rather than a wider demonstration — the export clamps to it and says how often it had to.

    Required, not defaulted, for the same reason as the closed width: a wrong default silently
    reshapes every exported width. Derive it with ``pingest calibrate-gripper``.

    :raises KeyError: if ``gripper_fingers.open_mm`` is missing from the config.
    """
    return float(load_gripper_calib()['gripper_fingers']['open_mm']) / 1000.0


def load_slam_config() -> dict:
    """
    Load SLAM step tunables from config/slam.yaml.

    Required, not optional: these values decide how much of each video ORB-SLAM3 ever
    sees, and an in-code default silently disagreeing with the checked-in file is how a
    corpus ends up half-processed at one stride and half at another. The file ships with
    the repo, so a missing one means a broken checkout, not a configuration choice.

    Raises:
        FileNotFoundError: if config/slam.yaml is absent or unreadable.

    """
    try:
        with SLAM_CONFIG_YAML.open() as f:
            return yaml.safe_load(f) or {}
    except OSError as err:
        raise FileNotFoundError(
            f'Cannot read {SLAM_CONFIG_YAML}, which is required to run the SLAM step '
            f'(it sets the resolution divisor and localization frame stride): {err}'
        ) from err
