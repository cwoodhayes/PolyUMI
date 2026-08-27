"""
Tests for the bringup-time payload push.

Every failure here is one fr3_bringup.launch.py aborts the whole launch on, so what matters is
that each way SetLoad can fail is actually reported as a failure — a rejected payload that returns
None would bring the arm up with a wrong gravity model, which is the exact thing the abort exists
to prevent.

Runs on the laptop despite targeting the Humble NUC: only the franka_msgs *message definitions*
are needed, and the service client is mocked, so no robot and no franka_bringup are involved.

    bash -c 'unset VIRTUAL_ENV; source /opt/ros/kilted/setup.bash \
      && source ros2_ws/install/setup.bash \
      && /usr/bin/python3 -m pytest nuc/test_set_payload.py -q'
"""

from unittest.mock import MagicMock, patch

import pytest
import rclpy

import set_payload
import tcp_calib


@pytest.fixture(scope='module', autouse=True)
def ros():
    """Init rclpy once for the module; the node here never joins a real executor."""
    rclpy.init()
    yield
    rclpy.shutdown()


def _node(*, service_up=True, response=MagicMock(success=True, error='')):
    """Build a node whose create_client hands back a mock service with the given outcome."""
    node = MagicMock()
    client = node.create_client.return_value
    client.wait_for_service.return_value = service_up
    client.call_async.return_value.done.return_value = True
    client.call_async.return_value.result.return_value = response
    return node


def test_request_carries_what_tcp_calib_measures():
    """The one place these numbers may come from is tcp_calib — not a second hardcoded copy."""
    req = set_payload._request()

    assert req.mass == pytest.approx(tcp_calib.PAYLOAD_MASS)
    assert list(req.center_of_mass) == pytest.approx(list(tcp_calib.payload_com_flange()))
    assert list(req.load_inertia) == pytest.approx(list(tcp_calib.payload_inertia_flange()))


def test_accepted_payload_reports_no_reason():
    """The success path is what lets fr3_arm_controller spawn, so it must not invent a failure."""
    with patch.object(set_payload.rclpy, 'spin_until_future_complete'):
        assert set_payload.set_payload(_node(), timeout_s=1.0) is None


def test_rejected_payload_is_a_failure_and_quotes_the_error():
    """
    `success: false` is the case `ros2 service call` reports as exit 0 — the reason this is a client.

    The error string is usually the useless "command exception error", but quoting it is what tells
    the operator to go read the /service_server log rather than re-run and hope.
    """
    rejected = MagicMock(success=False, error='command exception error')
    with patch.object(set_payload.rclpy, 'spin_until_future_complete'):
        reason = set_payload.set_payload(_node(response=rejected), timeout_s=1.0)

    assert reason is not None
    assert 'command exception error' in reason


def test_missing_service_is_a_failure_not_a_hang():
    """franka_bringup may still be coming up, but bringup blocks on this — so the wait is bounded."""
    reason = set_payload.set_payload(_node(service_up=False), timeout_s=1.0)

    assert reason is not None and set_payload.SERVICE in reason


def test_unanswered_call_is_a_failure():
    """A future that never resolves must not read as success — result() would be None."""
    node = _node()
    node.create_client.return_value.call_async.return_value.done.return_value = False
    with patch.object(set_payload.rclpy, 'spin_until_future_complete'):
        assert set_payload.set_payload(node, timeout_s=1.0) is not None
