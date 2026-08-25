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
#: Contact-mic step tunables: the per-frame block geometry that defines the exported
#: ``mic_0`` contract, and the diagnostic spectrogram's parameters.
CONTACT_AUDIO_YAML = INGEST_ROOT / 'config' / 'contact_audio.yaml'
#: Finger-camera export tunables: the crop that defines the exported ``finger_rgb`` contract,
#: an optional resize, and how stale a finger frame may be before an episode is refused.
FINGER_CAMERA_YAML = INGEST_ROOT / 'config' / 'finger_camera.yaml'


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


def _load_required_yaml(path: pathlib.Path, purpose: str) -> dict:
    """
    Load a checked-in YAML config, raising rather than falling back to an in-code default.

    Every caller here loads a data contract (block geometry, a crop, a resolution divisor) that
    every scene in a corpus must agree on — an in-code default silently disagreeing with the
    checked-in file is how a corpus ends up half-processed under one value and half under
    another, with nothing downstream able to tell. The file ships with the repo, so a missing
    one means a broken checkout, not a configuration choice.

    Raises:
        FileNotFoundError: if ``path`` is absent or unreadable.

    """
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except OSError as err:
        raise FileNotFoundError(f'Cannot read {path}, which is required to {purpose}: {err}') from err


def load_slam_config() -> dict:
    """Load SLAM step tunables (resolution divisor, localization frame stride) from config/slam.yaml."""
    return _load_required_yaml(SLAM_CONFIG_YAML, 'run the SLAM step')


def load_contact_audio_config() -> dict:
    """Load contact-mic step tunables (the mic_0 block geometry, log-mel params) from config/contact_audio.yaml."""
    return _load_required_yaml(CONTACT_AUDIO_YAML, 'run the contact-audio step')


def load_finger_camera_config() -> dict:
    """Load finger-camera export tunables (the finger_rgb crop, resize, staleness limit) from finger_camera.yaml."""
    return _load_required_yaml(FINGER_CAMERA_YAML, 'export the finger camera')
