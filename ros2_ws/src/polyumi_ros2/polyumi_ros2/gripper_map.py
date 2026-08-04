"""
The gripper-width contract between policy units and robot units.

The policy speaks in the units of the training data: the x-separation of the two ArUco finger
tags, as measured from GoPro video by ingest step 4 (``annotations/gripper_width/width_m``). That
is *not* jaw aperture — the tags sit on the outer faces of the fingers, so a fully-closed handheld
gripper still measures a few millimetres. The robot speaks in true aperture, where closed is 0.

The conversion is a **constant subtraction with slope 1**, following UMI. Their
``get_gripper_calibration_interpolator`` (``umi/common/interpolation_util.py``) computes
``aruco_actual_width - aruco_min_width``, and ``06_generate_dataset_plan.py`` passes the same
array as both of its arguments — so despite the ``interp1d`` machinery the map reduces to an
offset, with no rescaling. UMI measures that offset empirically as the minimum tag separation over
a calibration video in which the gripper fully closes (``scripts/calibrate_gripper_range.py``).

Two caveats worth knowing before trusting a number that comes out of here:

- **The offset PolyUMI uses is declared, not measured.** It defaults to ``gripper_calib.yaml``'s
  ``closed_mm``, a value no code has ever read. Writing our equivalent of UMI's calibration script
  is tracked in docs/franka-inference-bringup.md (Phase 2.5).
- **This aligns the zero point, not the stroke.** The handheld gripper can open wider than the
  Franka Hand's ~0.0817 m, so commands past ``max_width_m`` clamp — which shows up as the policy's
  intent saturating, not as an error.

The principled home for this subtraction is *ingest*, not inference: UMI applies it at
dataset-generation time, so its stored widths are already aperture and its inference path converts
nothing. Moving ours there is deferred to the exporter rework, since it invalidates existing
exports.
"""


def policy_to_robot_width(width_m: float, offset_m: float, max_width_m: float) -> float:
    """
    Convert a policy-space width (ArUco tag separation) to a commandable jaw aperture.

    :param width_m: width in training units, i.e. finger-tag separation in metres.
    :param offset_m: tag separation with the gripper fully closed; subtracted off.
    :param max_width_m: the robot's maximum aperture; the result is clamped into [0, this].
    :returns: jaw aperture in metres, clamped to the robot's reachable range.
    """
    return min(max(width_m - offset_m, 0.0), max_width_m)


def robot_to_policy_width(width_m: float, offset_m: float) -> float:
    """
    Convert a measured jaw aperture back into policy-space width.

    The inverse of :func:`policy_to_robot_width` up to clamping — deliberately *not* clamped here,
    because an out-of-range observation is real information about the robot and silently squashing
    it would hide a miscalibrated offset from the policy.

    :param width_m: jaw aperture in metres (for the FR3, ``position[0] + position[1]``).
    :param offset_m: tag separation with the gripper fully closed; added back on.
    :returns: width in training units, i.e. finger-tag separation in metres.
    """
    return width_m + offset_m
