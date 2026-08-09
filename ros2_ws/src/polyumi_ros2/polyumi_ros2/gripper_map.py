"""
The gripper-width contract between policy units and robot units.

The conversion is a **constant subtraction with slope 1**, following UMI. Their
``get_gripper_calibration_interpolator`` (``umi/common/interpolation_util.py``) computes
``aruco_actual_width - aruco_min_width``, and ``06_generate_dataset_plan.py`` passes the same array
as both of its arguments — so despite the ``interp1d`` machinery the map reduces to an offset, with
no rescaling. Slope 1 is right on geometry: ``tag_sep = aperture + const``, so a fitted slope would
only absorb the ArUco pipeline's scale error, which the training data already carries.

**What the policy's width means depends on when its checkpoint was exported**, and that decides
``offset_m``:

- Exports from 2026-08 onward: the DP exporter subtracts ``S_closed`` (see
  ``polyumi_ingest.config.load_closed_width_m``), so the policy speaks *opening from fully closed*,
  matching UMI. The correct offset is then ``-A_closed``.
- Earlier exports: the policy speaks *raw ArUco tag separation* — the x-separation of the two finger
  tags from ingest step 4, which is not aperture at all, since the tags sit on the fingers and a
  fully-closed gripper still measures several millimetres. The correct offset is
  ``S_closed - A_closed``.

A buffer records which it is as ``meta.attrs['gripper_closed_width_m']``.

``A_closed`` is the FR3's aperture with the fingers touching, which is **not zero** — the PolyUMI
fingers collide before the mechanism bottoms out. That is also why ``offset_m`` can legitimately
come out negative, and why the low clamp is ``min_width_m`` rather than 0.

Derive the three numbers with::

    pingest calibrate-gripper --scene <scene>     # S_closed, from an open/close recording
    ros2 run polyumi_ros2 gripper_range_probe     # A_closed and A_open, from the hand itself

**This aligns the zero point, not the stroke.** The handheld gripper opens wider than the Franka
Hand, so commands past ``max_width_m`` clamp — which shows up as the policy's intent saturating,
not as an error.
"""


def policy_to_robot_width(
    width_m: float, offset_m: float, max_width_m: float, min_width_m: float = 0.0
) -> float:
    """
    Convert a policy-space width to a commandable jaw aperture.

    :param width_m: width in training units — see the module docstring for which units those are,
        since it depends on when the checkpoint was exported.
    :param offset_m: subtracted off. May legitimately be **negative**; see the module docstring.
    :param max_width_m: the fingers' maximum reachable aperture (``A_open``).
    :param min_width_m: the fingers' minimum reachable aperture (``A_closed``). Not 0: the PolyUMI
        fingers collide before the mechanism bottoms out, and commanding below that point makes
        ``Move`` stall and abort — which is precisely what wedges ``fr3_gripper_bridge``'s deadband,
        since it records what a goal *accepted* rather than what the hand *reached*.
    :returns: jaw aperture in metres, clamped to the fingers' reachable range.
    """
    return min(max(width_m - offset_m, min_width_m), max_width_m)


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
