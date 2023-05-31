import cv2
import numpy as np

from src.utils.types import LandmarksArray


def bbox_from_landmarks(landmarks):
    X, Y = landmarks.T
    return (X.min(), Y.min(), X.max(), Y.max())


def mask_from_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """8 (or omitted) - 8-connected line.
          4 - 4-connected line.
    LINE_AA - antialiased line."""
    h, w = image.shape[:2]
    points = points.astype(int)
    assert points.shape[1] == 2, f"points.shape: {points.shape}"
    out = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(points.astype(int))
    cv2.fillConvexPoly(out, hull, 255, lineType=4)  # cv2.LINE_AA
    return out


def landmarks_68_symmetries():
    # 68 landmarks symmetry
    #
    sym_ids = [9, 58, 67, 63, 52, 34, 31, 30, 29, 28]
    sym = {
        1: 17,
        2: 16,
        3: 15,
        4: 14,
        5: 13,
        6: 12,
        7: 11,
        8: 10,
        #
        51: 53,
        50: 54,
        49: 55,
        60: 56,
        59: 57,
        #
        62: 64,
        61: 65,
        68: 66,
        #
        33: 35,
        32: 36,
        #
        37: 46,
        38: 45,
        39: 44,
        40: 43,
        41: 48,
        42: 47,
        #
        18: 27,
        19: 26,
        20: 25,
        21: 24,
        22: 23,
        #
        # id
        9: 9,
        58: 58,
        67: 67,
        63: 63,
        52: 52,
        34: 34,
        31: 31,
        30: 30,
        29: 29,
        28: 28,
    }
    return sym, sym_ids
