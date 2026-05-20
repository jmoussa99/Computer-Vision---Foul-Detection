"""Camera calibration and pose-estimation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def calibrate_from_chessboard(
    image_dir: str | Path,
    pattern_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
) -> dict[str, Any]:
    image_paths = sorted(Path(image_dir).glob("*"))
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_size
    object_points = []
    image_points = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        object_points.append(objp)
        image_points.append(refined)

    if not object_points or image_size is None:
        raise ValueError("No calibration chessboards were detected.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    return {
        "rms_reprojection_error": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "rotation_vectors": [r.tolist() for r in rvecs],
        "translation_vectors": [t.tolist() for t in tvecs],
        "images_used": len(object_points),
    }
