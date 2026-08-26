"""
Guard the two payload numbers tcp_calib computes rather than states.

Both fail invisibly. A mirrored yaw in the CoM conversion looks identical in the bringup log, the
service response and the robot's own inertia_load readback; a zero inertia tensor is accepted by
every layer of ROS and rejected only by the FR3 itself, as a bare "invalid argument".
"""

import math

import tcp_calib


def test_com_takes_the_flange_yaw_and_origin_from_hand_static_transforms(monkeypatch):
    """
    fr3_link8 -> fr3_hand is Rz(-45 deg) at some origin, and payload_com_flange applies both.

    +x in the hand must land at +x/-y in the flange (the sign is the half that fails silently),
    offset by the entry's own translation. That translation is zero today, so it is patched here
    rather than left to the shipped value — otherwise dropping the term again would still pass.
    """
    monkeypatch.setattr(
        tcp_calib,
        'HAND_STATIC_TRANSFORMS',
        (('fr3_link8', 'fr3_hand', (0.001, 0.002, 0.003), (0.0, 0.0, -math.pi / 4)),),
    )
    monkeypatch.setattr(tcp_calib, 'PAYLOAD_COM_HAND', (0.1, 0.0, 0.0))

    x, y, z = tcp_calib.payload_com_flange()

    assert math.isclose(x, 0.001 + 0.1 / math.sqrt(2))
    assert math.isclose(y, 0.002 - 0.1 / math.sqrt(2))
    assert math.isclose(z, 0.003)


def test_shipped_payload_constants_give_an_inertia_the_fr3_accepts():
    """
    The FR3 rejects mass>0 carrying a zero tensor as a bare "invalid argument".

    Asserted on the SHIPPED constants, not a patched mass: the tensor is a pure function of
    PAYLOAD_EXTENTS and PAYLOAD_MASS, so the only edit that can break it is an edit to those.
    """
    assert tcp_calib.PAYLOAD_MASS > 0
    assert all(v > 0 for v in tcp_calib.payload_inertia_flange()[::4])  # the three diagonal terms


def test_inertia_is_rotated_into_the_flange_not_left_in_the_hand():
    """
    SetLoad reads the tensor in the same frame as F_x_Cload, and the extents are a yaw away.

    A tensor left diagonal in fr3_hand is wrong in a way nothing downstream reports, so pin the
    two things the rotation must do: produce the xy term the hand frame does not have, and
    conserve the trace (a rotation cannot change it, but a botched one would).
    """
    fxx, fxy, fxz, fyx, fyy, fyz, fzx, fzy, fzz = tcp_calib.payload_inertia_flange()

    assert fxy != 0.0 and fxy == fyx, 'the -45 deg yaw must mix x and y, symmetrically'
    assert (fxz, fyz, fzx, fzy) == (0.0, 0.0, 0.0, 0.0), 'a yaw cannot tilt z out of plane'

    w, d, h = tcp_calib.PAYLOAD_EXTENTS
    k = tcp_calib.PAYLOAD_MASS / 12.0
    hand_trace = k * (d**2 + h**2) + k * (w**2 + h**2) + k * (w**2 + d**2)
    assert math.isclose(fxx + fyy + fzz, hand_trace)
