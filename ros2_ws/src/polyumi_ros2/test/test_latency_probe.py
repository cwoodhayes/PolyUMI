"""
Tests for the latency probe's parameter guards and signal handling.

The estimator itself is covered in test_latency_util; what is left here is everything that decides
whether the estimator gets fed something meaningful. Those are the quiet failures: a chirp that
aliases against the command rate, an amplitude the hand node's deadband swallows whole, or a
QR dedup that biases every offset upward. Each produces a plausible-looking number that then goes
into a robot config, so each gets a test.
"""

from unittest.mock import MagicMock

from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter

from polyumi_ros2.latency_probe import (
    CHIRP_BAND_HZ,
    COMMAND_HZ,
    MAX_BUFFERED_FRAMES,
    QR_QUIET_ZONE_MODULES,
    QR_RENDER_PX,
    LatencyProbe,
    decode_qr,
    first_stamp_per_code,
    render_qr,
)


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    """Init/shutdown rclpy once for the whole module."""
    rclpy.init()
    yield
    rclpy.shutdown()


def _probe(**overrides):
    """Build a probe with logging silenced, so a rejected construction still raises cleanly."""
    params = [Parameter(k, value=v) for k, v in overrides.items()]
    node = LatencyProbe(parameter_overrides=params)
    node.get_logger = MagicMock()
    return node


def test_unknown_mode_is_rejected():
    """A typo in mode must fail at construction, not silently pick a default measurement."""
    with pytest.raises(ValueError, match='mode must be one of'):
        _probe(mode='grippr')


def test_chirp_above_the_command_nyquist_is_rejected():
    """
    An aliased command sweep is the worst kind of failure: it correlates fine and means nothing.

    We record what we published, so if the published sweep aliases we would be comparing the arm's
    motion against a signal that was never really commanded.
    """
    with pytest.raises(ValueError, match='Nyquist'):
        _probe(mode='arm', chirp_f1_hz=5.0, command_hz=4.0)


def test_mode_defaults_stay_below_their_own_nyquist():
    """The shipped defaults must satisfy the guard they are checked against."""
    for mode, (f0, f1) in CHIRP_BAND_HZ.items():
        assert 0 < f0 < f1 < COMMAND_HZ[mode] / 2, mode


def test_gripper_step_must_clear_the_hand_deadband():
    """
    Below franka_hand_node's 5 mm deadband the step is discarded and the hand never moves.

    The run would then time out waiting for an onset, but only after the hardware slot is spent.
    """
    with pytest.raises(ValueError, match='deadband'):
        _probe(mode='gripper', step_m=0.004)


def test_gripper_step_cannot_be_centred_where_it_would_command_past_shut():
    """Stepping past fully closed means the hand stops on its stop, not where it was told."""
    with pytest.raises(ValueError, match='past shut'):
        _probe(mode='gripper', step_m=0.05, center_width_m=0.02)


def test_gripper_onset_threshold_must_sit_between_noise_and_the_step():
    """A threshold at or above half the step fires late; at zero it fires on nothing."""
    with pytest.raises(ValueError, match='onset_threshold_m'):
        _probe(mode='gripper', step_m=0.03, onset_threshold_m=0.02)


def test_chirp_sweeps_from_f0_to_f1_over_the_run():
    """
    The excitation must actually broaden, since bandwidth is what sharpens the correlation peak.

    Checked via zero crossings per half: a linear sweep puts more of them in the second half.
    """
    probe = _probe(mode='arm', duration_s=20.0)
    t = np.arange(0.0, 20.0, 0.001)
    x = np.asarray(probe._chirp(t))
    crossings = np.flatnonzero(np.diff(np.sign(x)))
    first_half = np.count_nonzero(crossings < len(t) / 2)
    assert first_half < (len(crossings) - first_half)
    assert np.abs(x).max() == pytest.approx(1.0, abs=0.01)
    probe.destroy_node()


def test_chirp_is_clamped_outside_the_run_window():
    """
    Past duration_s the sweep must hold, not keep accelerating.

    The publish loop is wall-clock driven, so the last iteration can land just past the end; an
    unclamped chirp would extrapolate to a frequency the plant cannot follow and inject a
    discontinuity right at the end of the record.
    """
    probe = _probe(mode='arm', duration_s=10.0)
    assert probe._chirp(10.5) == pytest.approx(probe._chirp(10.0))
    probe.destroy_node()


def test_gripper_state_is_summed_across_both_fingers():
    """Each FR3 finger reports half the aperture; the probe must record the full opening."""
    from sensor_msgs.msg import JointState

    probe = _probe(mode='gripper')
    msg = JointState()
    msg.header.stamp.sec = 7
    msg.header.stamp.nanosec = 500_000_000
    msg.position = [0.02, 0.021]
    probe._on_gripper_state(msg)
    assert [w for _, w in probe._actual] == [pytest.approx(0.041)]
    probe.destroy_node()


def test_gripper_state_is_timed_on_arrival_not_on_the_nuc_stamp():
    """
    The correlated series must be on ONE clock, and it is the laptop's.

    header.stamp comes from franka_hand_node on the NUC, while every instant it is compared
    against — _publish_width's return, _publish_chirp_chunk's anchor, _wait_until_still's window —
    is _now(). Correlating the two would put the laptop<->NUC clock offset straight into
    gripper_exec. The stamp is kept, but only for the publish interval.
    """
    from sensor_msgs.msg import JointState

    probe = _probe(mode='gripper')
    msg = JointState()
    # A stamp decades away from any plausible laptop clock, so using it cannot pass by accident.
    msg.header.stamp.sec = 7
    msg.header.stamp.nanosec = 500_000_000
    msg.position = [0.02, 0.021]
    before = probe._now()
    probe._on_gripper_state(msg)
    after = probe._now()

    assert before <= probe._actual[0][0] <= after
    assert probe._state_stamps == [pytest.approx(7.5)]
    probe.destroy_node()


def test_positionless_gripper_state_messages_are_ignored():
    """
    A state message carrying no position must not enter the series as a bogus aperture.

    A SINGLE position is not malformed — that is the FAULHABER driver's whole aperture, and
    requiring two silently dropped every one of its samples. See gripper_map.aperture_from_positions.
    """
    from sensor_msgs.msg import JointState

    probe = _probe(mode='gripper')
    msg = JointState()
    msg.position = []
    probe._on_gripper_state(msg)
    assert probe._actual == []

    msg.position = [0.02]
    probe._on_gripper_state(msg)
    assert [w for _, w in probe._actual] == [0.02]
    probe.destroy_node()


def test_gripper_onset_is_keyed_on_the_sample_stamp_not_the_polling_loop():
    """
    The onset must be keyed on a SAMPLE's own instant, not on the poll that happened to see it.

    Polling is deliberately faster than the 23 Hz state topic, so binding the answer to poll time
    would add a random fraction of a sample period to every rep. The instant is the sample's
    arrival rather than its NUC stamp — see _on_gripper_state — so it trails the fingers by the
    DDS hop, but t_command is on the same clock and the offset between them is not.
    """
    probe = _probe(mode='gripper', onset_threshold_m=0.001)
    probe._actual = [
        (10.0, 0.040),  # before the command; must be ignored even though it is far from rest
        (10.5, 0.0402),  # after the command but inside the threshold
        (10.6, 0.0455),  # first real motion
        (10.7, 0.050),
    ]
    assert probe._wait_for_onset(rest=0.040, t_command=10.2) == pytest.approx(10.6)
    probe.destroy_node()


def test_gripper_step_report_rejects_a_threshold_the_noise_reaches(tmp_path):
    """
    An onset threshold inside the encoder's own jitter triggers on nothing and reports ~0 ms.

    That is the failure that would silently zero latency.gripper_exec, so it must refuse to print
    a paste-able line rather than produce a confident small number.
    """
    probe = _probe(mode='gripper', onset_threshold_m=0.001, output_npz=str(tmp_path / 'g.npz'))
    lags = [('open', 0.3), ('close', 0.32), ('open', 0.28)]
    assert probe._report_gripper_steps(lags, worst_noise_m=0.0012) == 1
    assert probe._report_gripper_steps(lags, worst_noise_m=0.0001) == 0
    probe.destroy_node()


def test_gripper_report_emits_the_measured_latency_unmodified(tmp_path, capsys):
    """
    The gripper result goes into inference.yaml as measured, with no arithmetic against the arm.

    policy_client_node truncates each device's chunk by its own latency, so gripper_exec is a
    standalone number. It briefly was not: while the two devices shared one slice, the value had
    to be reported as (gripper - arm) / action_dt for the old bridge's gripper_lead_steps,
    which silently coupled it to a latency measured in a different run.
    """
    lags = [('open', 0.5), ('close', 0.54), ('open', 0.51)]  # median 0.51
    probe = _probe(mode='gripper', output_npz=str(tmp_path / 'a.npz'))
    probe._report_gripper_steps(lags, 0.0001)
    out = capsys.readouterr().out
    assert 'gripper_exec: 0.5100' in out
    assert 'gripper_lead_steps' not in out
    probe.destroy_node()


def test_gripper_chirp_amplitude_must_clear_the_hand_deadband():
    """
    Too small an amplitude leaves too few deadband-sized steps for the sweep to read as a sinusoid.

    franka_hand_node suppresses any width within its deadband of the last one it sent, so the
    hand is only ever offered a staircase. The quantisation is in SPACE, not time — how many steps
    fit in a half stroke depends on amplitude alone, which is why frequency does not appear here.
    """
    with pytest.raises(ValueError, match='deadband-sized steps'):
        _probe(mode='gripper_chirp', amplitude_m=0.005)


def test_gripper_chirp_is_bounded_by_the_hand_command_floor_not_the_publish_rate():
    """
    Publishing faster than the hand can act buys nothing; above its Nyquist the sweep cannot land.

    The guard against command_hz alone would wave this through — 2.5 Hz is well under the 10 Hz
    publish Nyquist of 5 Hz — while the hand, completing one Move per HAND_PERIOD_S, gets fewer
    than two per cycle. That is a sweep it could not reproduce even if it were perfect.
    """
    with pytest.raises(ValueError, match="Nyquist of the hand's own"):
        _probe(mode='gripper_chirp', chirp_f1_hz=2.5)


def test_gripper_chirp_defaults_satisfy_their_own_guards():
    """
    The shipped band and amplitude must construct. Guards this file's constants against each other.

    This is the test that catches a hand-tuned CHIRP_BAND_HZ entry that the probe would then refuse
    to run at all — which is otherwise only discovered with the hardware already booked.
    """
    probe = _probe(mode='gripper_chirp')
    probe.destroy_node()


def test_gripper_chirp_publishes_an_absolutely_timed_chunk():
    """
    Each publication is a full n_action_steps chunk whose waypoints name absolute instants.

    A single already-due waypoint is what franka_hand_node can only ever chase — it is the shape
    that made this mode report 0.00 mm of travel on hardware. The chunk is what lets the node's
    scheduling branch run, so its depth and its timing are the point of the mode.
    """
    probe = _probe(mode='gripper_chirp', n_action_steps=5, lead_s=0.3, action_dt=0.1)
    published = []
    probe._pub = SimpleNamespace(publish=published.append, get_subscription_count=lambda: 1)

    due_s, width = probe._publish_chirp_chunk(0.0)

    assert len(published) == 1
    msg = published[0]
    assert len(msg.points) == 5
    stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    # The anchor leads publication, and the reported instant is the first waypoint's, not now.
    assert due_s == pytest.approx(stamp_s)
    assert width == pytest.approx(msg.points[0].positions[0])
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in msg.points]
    assert times == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    # Sampled at the instant each waypoint is due, so the chunk describes the swept target.
    assert msg.points[1].positions[0] == pytest.approx(probe._center + probe._amplitude * float(probe._chirp(0.4)))
    probe.destroy_node()


def test_gripper_chirp_rejects_a_chunk_that_can_never_be_scheduled():
    """lead_s must be positive, or every chunk is due the moment it is published."""
    with pytest.raises(ValueError, match='lead_s must be > 0'):
        _probe(mode='gripper_chirp', lead_s=0.0)
    with pytest.raises(ValueError, match='n_action_steps must be >= 1'):
        _probe(mode='gripper_chirp', n_action_steps=0)


def test_gripper_chirp_cannot_command_past_the_hand_stroke():
    """An amplitude that would push the sweep past 0 or HAND_MAX_WIDTH_M must be rejected."""
    with pytest.raises(ValueError, match='must stay within'):
        _probe(mode='gripper_chirp', amplitude_m=0.05)


def test_gripper_chirp_recovers_a_known_lag_end_to_end(tmp_path, capsys):
    """
    Feed the xcorr reporter a commanded chirp and a copy delayed by a known lag; check it comes back.

    This is the one check that fails if the new mode's wiring into _report_xcorr is wrong — the
    estimator itself is already covered by test_latency_util.
    """
    true_lag = 0.65
    t = np.arange(0.0, 20.0, 1 / 50.0)
    probe = _probe(mode='gripper_chirp', output_npz=str(tmp_path / 'gc.npz'), plot=False)
    commanded = [(float(ti), 0.04 + 0.03 * float(probe._chirp(ti))) for ti in t]
    actual = [(ti + true_lag, w) for ti, w in commanded]
    # A separate series from `actual` above — these are the NUC-side publish stamps, which is what
    # _joint_state_interval_lines summarises. 50 ms apart, so latency.gripper is half of that.
    probe._state_stamps = [float(i) * 0.05 for i in range(20)]

    assert probe._report_xcorr('gripper_exec', commanded, actual, extra_lines=probe._joint_state_interval_lines()) == 0
    saved = np.load(tmp_path / 'gc.npz')
    assert float(saved['latency_s']) == pytest.approx(true_lag, abs=0.02)
    out = capsys.readouterr().out
    # Against the saved value rather than a literal, so re-tuning the shipped band cannot break
    # this on a rounding difference that says nothing about the wiring.
    assert f'gripper_exec: {float(saved["latency_s"]):.4f}' in out
    assert 'latency.gripper: 0.0250' in out
    probe.destroy_node()


def _filmed(qr, blur=0, perspective=0, scale=1.0):
    """Paste a rendered QR into a 1080p frame and degrade it the way filming a screen would."""
    import cv2

    size = int(qr.shape[0] * scale)
    frame = np.full((1080, 1920), 240, np.uint8)
    y, x = (1080 - size) // 2, (1920 - size) // 2
    frame[y : y + size, x : x + size] = cv2.resize(qr, (size, size))
    if perspective:
        p = perspective
        matrix = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080]]),
            np.float32([[p, p * 0.5], [1920 - p, 0], [1920, 1080 - p], [p * 0.3, 1080]]),
        )
        frame = cv2.warpPerspective(frame, matrix, (1920, 1080), borderValue=240)
    if blur:
        frame = cv2.GaussianBlur(frame, (blur | 1, blur | 1), 0)
    return frame


@pytest.mark.parametrize(
    'label,kwargs',
    [
        ('head on', {}),
        ('soft focus', {'blur': 11}),
        ('off axis', {'perspective': 60}),
        ('off axis, soft focus', {'perspective': 40, 'blur': 5}),
        ('screen fills less of the frame', {'scale': 0.45}),
    ],
)
def test_qr_survives_being_filmed(label, kwargs):
    """
    Render and read a code back under the conditions a handheld GoPro actually produces.

    This is the only part of mode:=camera testable without hardware, and it earns its place: the
    decoder upstream UMI uses (``detectAndDecodeCurved``) reads *none* of these on OpenCV 4.6,
    including the head-on case. Ported verbatim it would have returned zero codes on the rig and
    looked like bad aim. Guard the substitution so nobody swaps it back.
    """
    import cv2

    payload = '1754800000.123456'
    frame = _filmed(render_qr(cv2.QRCodeEncoder.create(), payload), **kwargs)
    assert decode_qr(cv2.QRCodeDetector(), frame) == payload, label


def test_decode_returns_empty_for_a_frame_with_no_code():
    """Most frames of a run show no code; that must be an empty string, not an exception."""
    import cv2

    blank = np.full((1080, 1920), 200, np.uint8)
    assert decode_qr(cv2.QRCodeDetector(), blank) == ''


def test_rendered_qr_carries_a_quiet_zone():
    """
    Without the 4-module border no decoder finds the finder patterns; cv2's encoder omits it.

    Checked structurally rather than via a decode, so a failure says which of the two things broke.
    """
    import cv2

    qr = render_qr(cv2.QRCodeEncoder.create(), '1.0')
    border = max(1, round(QR_QUIET_ZONE_MODULES * QR_RENDER_PX / qr.shape[0]))
    assert (qr[:border, :] == 255).all()
    assert (qr[:, :border] == 255).all()


def test_camera_pipeline_recovers_a_known_latency_end_to_end(tmp_path):
    """
    Feed the camera reporting path frames whose true lag is known, and check the number comes back.

    This is the closest thing to a hardware test for the most load-bearing constant in the system:
    QR render -> filmed degradation -> decode -> dedup -> render-overhead subtraction -> mean. Each
    stage is individually plausible and wrong end-to-end is exactly the failure that would ship a
    bad latency.gopro. Ground truth is free here, so there is no excuse not to check it.
    """
    import cv2

    true_lag, render_overhead = 0.150, 0.010
    encoder = cv2.QRCodeEncoder.create()
    probe = _probe(mode='camera', output_npz=str(tmp_path / 'cam.npz'), plot=False)
    # 20 Hz of codes, each filmed by three 60 fps frames; only the first of the three is a
    # measurement, so a broken dedup shows up as a latency biased by ~half a QR period.
    for i in range(15):
        qr_time = 1_754_800_000.0 + i / 20.0
        frame = _filmed(render_qr(encoder, f'{qr_time:.6f}'), blur=3, scale=0.8)
        for repeat in range(3):
            stamp = qr_time + true_lag + repeat / 60.0
            probe._frames.append((stamp, stamp + 0.2, frame))

    assert probe._report_camera(render_overhead) == 0
    saved = np.load(tmp_path / 'cam.npz')
    assert float(saved['latency_s']) == pytest.approx(true_lag - render_overhead, abs=0.002)
    probe.destroy_node()


def test_camera_render_overhead_is_the_draw_only_not_the_qr_pacing(monkeypatch):
    """
    The subtracted overhead must be the draw, not the draw plus the wait between codes.

    This shipped wrong: the sample was taken after the single ``waitKey(1/QR_HZ)`` that both
    painted the window and paced the loop, so a whole QR period — 50 ms, a third of latency.gopro
    itself — was subtracted from every offset. The resulting 0.0756 s sat well below UMI's
    0.125-0.17 s band for the same capture chain, which is the only reason it was caught. Nothing
    about the number looks wrong on its own, so the guard has to be here.
    """
    import time

    from polyumi_ros2 import latency_probe as lp

    probe = _probe(mode='camera', duration_s=1.05, plot=False)
    for name in ('namedWindow', 'setWindowProperty', 'destroyAllWindows', 'imshow'):
        monkeypatch.setattr(lp.cv2, name, lambda *a, **k: None)
    # A waitKey that actually sleeps, which is what makes the difference observable at all.
    monkeypatch.setattr(lp.cv2, 'waitKey', lambda ms: (time.sleep(ms / 1e3), -1)[1])
    subtracted = []
    monkeypatch.setattr(probe, '_report_camera', lambda o: (subtracted.append(o), 0)[1])

    assert probe.run() == 0
    # Generous: the draw is ~1 ms plus the QR encode, against a 50 ms period. The old code landed
    # at ~50 ms and anything near that is the bug back.
    assert subtracted[0] < 0.5 / lp.QR_HZ, f'{subtracted[0] * 1e3:.0f} ms looks like the QR period'
    probe.destroy_node()


def test_camera_reports_failure_when_too_few_codes_decode():
    """A blurred or badly aimed run must refuse to produce a number rather than average three."""
    probe = _probe(mode='camera', plot=False)
    blank = np.full((480, 640), 200, np.uint8)
    for i in range(10):
        probe._frames.append((float(i), float(i) + 0.2, blank))
    assert probe._report_camera(0.01) == 1
    probe.destroy_node()


def test_camera_buffer_stops_accepting_frames_once_full():
    """
    The cap bounds RAM, and the display loop reads it to stop rather than discard silently.

    At 60 fps the cap is reached in a few seconds, well before the default duration_s — so without
    the early stop the probe would keep showing codes that no frame in the record ever saw.
    """
    from sensor_msgs.msg import Image

    probe = _probe(mode='camera')
    msg = Image()
    msg.height, msg.width, msg.encoding = 4, 4, 'rgb8'
    msg.data = bytes(4 * 4 * 3)
    assert not probe.frames_full()
    for _ in range(MAX_BUFFERED_FRAMES + 10):
        probe._on_image(msg)
    assert probe.frames_full()
    assert len(probe._frames) == MAX_BUFFERED_FRAMES
    probe.destroy_node()


def test_camera_rejects_encodings_it_cannot_index():
    """Same contract as policy_client_node: reshaping a non-3-channel buffer as one would crash."""
    from sensor_msgs.msg import Image

    probe = _probe(mode='camera')
    msg = Image()
    msg.height, msg.width, msg.encoding = 4, 4, 'yuv422_yuy2'
    msg.data = bytes(4 * 4 * 2)
    probe._on_image(msg)
    assert probe._frames == []
    probe.destroy_node()


def test_qr_dedup_keeps_the_earliest_frame_that_saw_each_code():
    """
    A code shown at 20 Hz is filmed by several 60 fps frames; only the first is a measurement.

    Averaging all of them would bias every offset upward by roughly half a QR period — 25 ms at
    20 Hz, the same order as the difference between UMI's 0.125 and 0.17.
    """
    decoded = [(1.000, '100.0'), (1.017, '100.0'), (1.033, '100.0'), (1.050, '100.05')]
    assert first_stamp_per_code(decoded) == {100.0: 1.000, 100.05: 1.050}


def test_qr_dedup_ignores_frames_that_decoded_to_nothing_or_to_junk():
    """Most frames decode to an empty string; a garbled one must not become a float() crash."""
    assert first_stamp_per_code([(1.0, ''), (1.1, 'not-a-time'), (1.2, '5.0')]) == {5.0: 1.2}


def test_tf_sampler_drops_repeated_stamps():
    """
    TF is polled faster than the arm publishes, so most polls return the same transform again.

    Letting duplicates through would pile many grid samples onto one instant and skew the
    resampling toward whatever the arm was doing then.
    """
    probe = _probe(mode='arm')
    tf = MagicMock()
    tf.header.stamp.sec = 3
    tf.header.stamp.nanosec = 0
    tf.transform.translation.x = 0.1
    tf.transform.translation.y = 0.2
    tf.transform.translation.z = 0.3
    probe._tf_buffer = MagicMock(lookup_transform=MagicMock(return_value=tf))

    probe._sample_tcp(1)
    probe._sample_tcp(1)
    assert probe._actual == [(3.0, pytest.approx(0.2))]

    tf.header.stamp.nanosec = 10_000_000
    probe._sample_tcp(1)
    assert len(probe._actual) == 2
    probe.destroy_node()
