"""
The gripper-width contract between policy units and robot units.

The conversion is a **constant subtraction with slope 1**, following UMI. Their
``get_gripper_calibration_interpolator`` (``umi/common/interpolation_util.py``) computes
``aruco_actual_width - aruco_min_width``, and ``06_generate_dataset_plan.py`` passes the same array
as both of its arguments — so despite the ``interp1d`` machinery the map reduces to an offset, with
no rescaling. Slope 1 is right on geometry: ``tag_sep = aperture + const``, so a fitted slope would
only absorb the ArUco pipeline's scale error, which the training data already carries.

The policy speaks **metres of opening from fully closed**: the DP exporter subtracts ``S_closed``
(see ``polyumi_ingest.config.load_closed_width_m``) from step 4's raw ArUco tag separation, so
exported widths are already a physical opening rather than a tag measurement. A buffer records the
value it was built with as ``meta.attrs['gripper_closed_width_m']``.

The offset is then ``-A_closed``, the FR3's aperture with the fingers touching. **Measured
2026-08-09 that is 0.0**, so this map is currently the identity plus clamping — these fingers meet
at the mechanism's true zero, which puts us in the same position as UMI's WSG, whose inference path
likewise converts nothing. **With the current fingers nothing collides early at either end** — the
hand's own limits are the binding ones. A redesign whose tips met before the mechanism bottomed out
would make ``A_closed`` non-zero and the offset negative; that is legitimate and the validation
permits it, which is why the low clamp is ``min_width_m`` rather than a hardcoded 0.

Derive the numbers with::

    pingest calibrate-gripper --scene <scene>     # S_closed, from an open/close recording
    ros2 run polyumi_ros2 gripper_range_probe     # A_closed and A_open, from the hand itself

Checkpoints exported before 2026-08-09 speak raw tag separation instead and would need
``offset_m = S_closed - A_closed``. Support for them was dropped rather than carried; the value is
recorded here only so an old buffer can be identified from its missing metadata attr.

**This aligns the zero point, not the stroke**, and the two mechanisms genuinely differ: the
handheld reaches 132.3 mm of tag separation where the FR3 manages 126.2 mm, so the top ~7% of the
policy's commanded range clamps at ``max_width_m``. That surfaces as the policy's intent saturating
rather than as an error.
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
    :param min_width_m: the fingers' minimum reachable aperture (``A_closed``). **0.0 with the
        current fingers**, which meet exactly at the mechanism's zero — so this defaults to 0 and
        the parameter is presently inert. It exists because fingers whose tips collided before the
        mechanism bottomed out would make it non-zero, and commanding below that point makes
        ``Move`` stall and abort — which is what wedges ``fr3_gripper_bridge``'s deadband, since it
        records what a goal *accepted* rather than what the hand *reached*.
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
