"""Tests for SlamToWorldAlignStep and _svd_align."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from polyumi_ingest.preproc.so_align_step import _svd_align


def _make_transform(euler_deg: tuple[float, float, float], translation: tuple[float, float, float]):
    """Return (R, t) for a rotation specified as extrinsic XYZ Euler angles."""
    R = Rotation.from_euler('xyz', euler_deg, degrees=True).as_matrix()
    t = np.array(translation)
    return R, t


def _apply(R, t, pts):
    return (R @ pts.T).T + t


class TestSvdAlign:
    def test_identity(self):
        rng = np.random.default_rng(0)
        src = rng.uniform(-1, 1, (20, 3))
        R_est, t_est = _svd_align(src, src.copy())
        assert np.allclose(R_est, np.eye(3), atol=1e-10)
        assert np.allclose(t_est, 0, atol=1e-10)

    def test_pure_translation(self):
        rng = np.random.default_rng(1)
        src = rng.uniform(-1, 1, (20, 3))
        t_true = np.array([5.0, -3.0, 1.5])
        dst = src + t_true
        R_est, t_est = _svd_align(src, dst)
        assert np.allclose(R_est, np.eye(3), atol=1e-10)
        assert np.allclose(t_est, t_true, atol=1e-10)

    def test_known_rotation_and_translation(self):
        rng = np.random.default_rng(42)
        R_true, t_true = _make_transform((30.0, -15.0, 90.0), (1.0, 2.0, 3.0))
        src = rng.uniform(-2, 2, (100, 3))
        dst = _apply(R_true, t_true, src)

        R_est, t_est = _svd_align(src, dst)

        angle_err = np.degrees(np.arccos(np.clip((np.trace(R_true.T @ R_est) - 1) / 2, -1.0, 1.0)))
        assert angle_err < 1e-4, f'Rotation error {angle_err:.2e} deg'
        assert np.allclose(t_est, t_true, atol=1e-8), f't error {np.linalg.norm(t_est - t_true):.2e}'

    def test_no_reflection(self):
        """det(R) must be +1 (rotation, not improper rotation)."""
        rng = np.random.default_rng(7)
        R_true, t_true = _make_transform((45.0, 20.0, -10.0), (0.5, -1.0, 2.0))
        src = rng.uniform(-1, 1, (50, 3))
        dst = _apply(R_true, t_true, src)
        R_est, _ = _svd_align(src, dst)
        assert abs(np.linalg.det(R_est) - 1.0) < 1e-10

    def test_noisy_correspondence(self):
        """With Gaussian noise on correspondences, error should still be small."""
        rng = np.random.default_rng(99)
        R_true, t_true = _make_transform((10.0, 20.0, 30.0), (0.1, 0.2, 0.3))
        src = rng.uniform(-1, 1, (200, 3))
        noise = rng.normal(0, 0.005, src.shape)
        dst = _apply(R_true, t_true, src) + noise

        R_est, t_est = _svd_align(src, dst)

        angle_err = np.degrees(np.arccos(np.clip((np.trace(R_true.T @ R_est) - 1) / 2, -1.0, 1.0)))
        assert angle_err < 0.5, f'Rotation error too large: {angle_err:.2f} deg'
        assert np.linalg.norm(t_est - t_true) < 0.01, f'Translation error too large'

    def test_minimum_points(self):
        """Should work with as few as 3 non-collinear points."""
        R_true, t_true = _make_transform((0.0, 0.0, 45.0), (1.0, 0.0, 0.0))
        src = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        dst = _apply(R_true, t_true, src)
        R_est, t_est = _svd_align(src, dst)
        angle_err = np.degrees(np.arccos(np.clip((np.trace(R_true.T @ R_est) - 1) / 2, -1.0, 1.0)))
        assert angle_err < 1e-5
        assert np.allclose(t_est, t_true, atol=1e-10)
