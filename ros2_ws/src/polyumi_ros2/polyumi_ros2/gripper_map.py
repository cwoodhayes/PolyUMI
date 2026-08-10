"""
The gripper-width contract between policy units and robot units.

Three measured quantities are referred to throughout, all PolyUMI's own:

* the **closed width** — ArUco finger-tag separation with the handheld gripper's fingers touching.
  Lives in ``ingest/config/gripper_calib.yaml`` as ``closed_mm``.
* the **closed aperture** — the FR3's jaw aperture in that same physical configuration.
  Here as ``min_width_m`` (``gripper_min_width_m`` in ``config/inference.yaml``).
* the **open aperture** — the FR3's jaw aperture at full open. Here as ``max_width_m``.

The conversion is a **constant subtraction with slope 1**, following UMI. In *their* repo
(upstream ``universal_manipulation_interface``, checked out alongside this one),
``get_gripper_calibration_interpolator`` in ``umi/common/interpolation_util.py`` computes
``aruco_actual_width - aruco_min_width``, and ``scripts_slam_pipeline/06_generate_dataset_plan.py``
passes the same array as both of its arguments — so despite the ``interp1d`` machinery the map
reduces to an offset, with no rescaling. Every filename in that sentence is theirs, not ours.
Slope 1 is right on geometry: ``tag_sep = aperture + const``, so a fitted slope would only absorb
the ArUco pipeline's scale error, which the training data already carries.

The policy speaks **metres of opening from fully closed**: the DP exporter subtracts the closed
width (see ``polyumi_ingest.config.load_closed_width_m``) from step 4's raw ArUco tag separation,
so exported widths are already a physical opening rather than a tag measurement. A buffer records
the value it was built with as ``meta.attrs['gripper_closed_width_m']``.

So the arm-side conversion is just **add the closed aperture back on**, and there is deliberately
no separate offset parameter: policy width 0 means "fully closed", and fully closed on the arm is
the closed aperture, so the offset and ``min_width_m`` are the same measured number. There used to
be a ``gripper_offset_m`` alongside it, which was redundant by construction — one measurement
behind two knobs of opposite sign, and setting only one of them was wrong by that amount across
the whole range while still reading correctly at the closed end, where the low clamp hid it.

**Measured 2026-08-09 the closed aperture is 0.0**, so this map is currently the identity plus
clamping — these fingers meet at the mechanism's true zero, which puts us in the same position as
UMI's WSG, whose inference path likewise converts nothing. **With the current fingers nothing
collides early at either end**; the hand's own limits are the binding ones. A redesign whose tips
met before the mechanism bottomed out would make it non-zero, which is why the low clamp is
``min_width_m`` rather than a hardcoded 0.

Derive the numbers with::

    pingest calibrate-gripper --scene <scene>   # the closed width, from an open/close recording
    ros2 run polyumi_ros2 gripper_range_probe   # both apertures, from the hand itself

The closed width does not appear in this file at all — the DP exporter has already subtracted it
by the time a width reaches the policy. It only mattered to inference for checkpoints exported
before 2026-08-09, which spoke raw tag separation and needed ``closed_width - closed_aperture``
here; support for those was dropped rather than carried.

**This aligns the zero point, not the stroke**, and the two mechanisms genuinely differ: the
handheld reaches 132.3 mm of tag separation where the FR3 manages 126.2 mm, so the top ~7% of the
policy's commanded range clamps at ``max_width_m``. That surfaces as the policy's intent saturating
rather than as an error.
"""


def policy_to_robot_width(width_m: float, min_width_m: float, max_width_m: float) -> float:
    """
    Convert a policy-space width to a commandable jaw aperture.

    The offset *is* ``min_width_m``, which is why there is no separate offset parameter: policy
    width 0 means "fully closed", and fully closed on the arm is the closed aperture. Carrying the
    two independently let them be set inconsistently — one measurement, two knobs, opposite signs —
    and the failure was silent, because the low clamp masks it at exactly the one width anybody
    would spot-check.

    :param width_m: metres of opening from fully closed, the DP exporter's convention.
    :param min_width_m: the fingers' minimum reachable aperture, i.e. the closed aperture. 0.0 with
        the current fingers, which meet at the mechanism's zero. Fingers whose tips collided first
        would make it non-zero, and commanding below that point makes ``Move`` stall and abort —
        which is what wedges ``fr3_gripper_bridge``'s deadband, since it records what a goal
        *accepted* rather than what the hand *reached*.
    :param max_width_m: the fingers' maximum reachable aperture, i.e. the open aperture.
    :returns: jaw aperture in metres, clamped to the fingers' reachable range.
    """
    return min(max(width_m + min_width_m, min_width_m), max_width_m)


def robot_to_policy_width(width_m: float, min_width_m: float) -> float:
    """
    Convert a measured jaw aperture back into policy-space width.

    The inverse of :func:`policy_to_robot_width` up to clamping — deliberately *not* clamped here,
    because an out-of-range observation is real information about the robot and silently squashing
    it would hide a miscalibrated closed aperture from the policy.

    :param width_m: jaw aperture in metres (for the FR3, ``position[0] + position[1]``).
    :param min_width_m: the closed aperture, subtracted off to reach opening-from-closed.
    :returns: metres of opening from fully closed, the units the policy was trained in.
    """
    return width_m - min_width_m
