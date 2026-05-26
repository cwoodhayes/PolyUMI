"""
Verbatim helpers copied from the original UMI repo.

Source: https://github.com/real-stanford/universal_manipulation_interface
Files: umi/common/cv_util.py and umi/common/interpolation_util.py

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
Licensed under the MIT License. The original license is reproduced below.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

Function bodies are copied unchanged so that behaviour matches the original
UMI ArUco-based gripper-width estimation pipeline. Type hints, formatting, and
imports are kept consistent with the upstream source.

Two adaptations inside ``detect_localize_aruco_tags`` for OpenCV >= 4.7 (we run
on 4.13), neither of which changes behaviour:

* ``cv2.aruco.detectMarkers`` (removed) → ``cv2.aruco.ArucoDetector.detectMarkers``.
* ``cv2.aruco.estimatePoseSingleMarkers`` (removed) → ``cv2.solvePnP`` with
  ``SOLVEPNP_IPPE_SQUARE`` and the marker's canonical square object points
  in the same corner order detectMarkers returns.
"""

# ruff: noqa
# fmt: off
from typing import Dict, Tuple

import copy
import cv2
import numpy as np
import scipy.interpolate as si


# =================== intrinsics ===================

def parse_fisheye_intrinsics(json_data: dict) -> Dict[str, np.ndarray]:
    """
    Reads camera intrinsics from OpenCameraImuCalibration to opencv format.
    Example:
    {
        "final_reproj_error": 0.17053819312281043,
        "fps": 60.0,
        "image_height": 1080,
        "image_width": 1920,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": 1.0026582765352035,
            "focal_length": 420.56809123853304,
            "principal_pt_x": 959.857586309181,
            "principal_pt_y": 542.8155851051391,
            "radial_distortion_1": -0.011968137016185161,
            "radial_distortion_2": -0.03929790706019372,
            "radial_distortion_3": 0.018577224235396064,
            "radial_distortion_4": -0.005075629959840777,
            "skew": 0.0
        },
        "nr_calib_images": 129,
        "stabelized": false
    }
    """
    assert json_data['intrinsic_type'] == 'FISHEYE'
    intr_data = json_data['intrinsics']

    # img size
    h = json_data['image_height']
    w = json_data['image_width']

    # pinhole parameters
    f = intr_data['focal_length']
    px = intr_data['principal_pt_x']
    py = intr_data['principal_pt_y']

    # Kannala-Brandt non-linear parameters for distortion
    kb8 = [
        intr_data['radial_distortion_1'],
        intr_data['radial_distortion_2'],
        intr_data['radial_distortion_3'],
        intr_data['radial_distortion_4']
    ]

    opencv_intr_dict = {
        'DIM': np.array([w, h], dtype=np.int64),
        'K': np.array([
            [f, 0, px],
            [0, f, py],
            [0, 0, 1]
        ], dtype=np.float64),
        'D': np.array([kb8]).T
    }
    return opencv_intr_dict


def convert_fisheye_intrinsics_resolution(
        opencv_intr_dict: Dict[str, np.ndarray],
        target_resolution: Tuple[int, int]
        ) -> Dict[str, np.ndarray]:
    """
    Convert fisheye intrinsics parameter to a different resolution,
    assuming that images are not cropped in the vertical dimension,
    and only symmetrically cropped/padded in horizontal dimension.
    """
    iw, ih = opencv_intr_dict['DIM']
    iK = opencv_intr_dict['K']
    ifx = iK[0,0]
    ify = iK[1,1]
    ipx = iK[0,2]
    ipy = iK[1,2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array([
        [ofx, 0, opx],
        [0, ofy, opy],
        [0, 0, 1]
    ], dtype=np.float64)

    out_intr_dict = copy.deepcopy(opencv_intr_dict)
    out_intr_dict['DIM'] = np.array([ow, oh], dtype=np.int64)
    out_intr_dict['K'] = oK
    return out_intr_dict


# ================= ArUcO tag =====================

def detect_localize_aruco_tags(
        img: np.ndarray,
        aruco_dict: cv2.aruco.Dictionary,
        marker_size_map: Dict[int, float],
        fisheye_intr_dict: Dict[str, np.ndarray],
        refine_subpix: bool=True):
    K = fisheye_intr_dict['K']
    D = fisheye_intr_dict['D']
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # OpenCV >= 4.7: cv2.aruco.detectMarkers is gone; use ArucoDetector instead.
    detector = cv2.aruco.ArucoDetector(aruco_dict, param)
    corners, ids, rejectedImgPoints = detector.detectMarkers(img)
    if ids is None or len(corners) == 0:
        return dict()

    tag_dict = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue

        marker_size_m = marker_size_map[this_id]
        undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)
        # cv2.aruco.estimatePoseSingleMarkers was removed in OpenCV >= 4.7.
        # Replicate it with solvePnP and the canonical marker object points
        # in the order returned by detectMarkers (top-left, top-right,
        # bottom-right, bottom-left).
        half = marker_size_m / 2.0
        obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)
        img_points = undistorted.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, K, np.zeros((1,5)),
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        tag_dict[this_id] = {
            'rvec': rvec.squeeze(),
            'tvec': tvec.squeeze(),
            'corners': this_corners.squeeze()
        }
    return tag_dict


def get_gripper_width(tag_dict, left_id, right_id, nominal_z=0.072, z_tolerance=0.008):
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]['tvec']
        # check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]['tvec']
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width


# =================== interpolation ===================

def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x,
        axis=0, bounds_error=False,
        fill_value=(x[0], x[-1]))
    return gripper_interp
