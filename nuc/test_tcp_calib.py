"""
Guard the two payload numbers tcp_calib computes rather than states.

Both fail invisibly. A mirrored yaw in the CoM conversion looks identical in the bringup log, the
service response and the robot's own inertia_load readback; a zero inertia tensor is accepted by
every layer of ROS and rejected only by the FR3 itself, as a bare "invalid argument".
"""

import math

import tcp_calib


def test_on_axis_com_is_unchanged_by_the_flange_yaw(monkeypatch):
    """A CoM on the approach axis sits on the rotation axis, so the yaw must not move it."""
    monkeypatch.setattr(tcp_calib, 'PAYLOAD_COM_HAND', (0.0, 0.0, 0.09))
    x, y, z = tcp_calib.payload_com_flange()
    assert (round(x, 12), round(y, 12), z) == (0.0, 0.0, 0.09)


def test_off_axis_com_rotates_by_minus_45_degrees(monkeypatch):
    """fr3_link8 -> fr3_hand is Rz(-45 deg), so +x in the hand lands at +x/-y in the flange."""
    monkeypatch.setattr(tcp_calib, 'PAYLOAD_COM_HAND', (0.1, 0.0, 0.0))
    x, y, z = tcp_calib.payload_com_flange()
    assert math.isclose(x, 0.1 / math.sqrt(2))
    assert math.isclose(y, -0.1 / math.sqrt(2))
    assert z == 0.0


def test_nonzero_mass_never_yields_a_zero_inertia(monkeypatch):
    """The FR3 rejects mass>0 with a zero tensor as "invalid argument", so guard the pairing."""
    monkeypatch.setattr(tcp_calib, 'PAYLOAD_MASS', 0.55)
    assert all(v > 0 for v in tcp_calib.payload_inertia()[::4])  # the three diagonal terms


def test_inertia_satisfies_the_triangle_inequality(monkeypatch):
    """Principal moments of a real body obey Ixx + Iyy >= Izz; the firmware checks this."""
    monkeypatch.setattr(tcp_calib, 'PAYLOAD_MASS', 0.55)
    ixx, iyy, izz = tcp_calib.payload_inertia()[::4]
    assert ixx + iyy >= izz and iyy + izz >= ixx and izz + ixx >= iyy
