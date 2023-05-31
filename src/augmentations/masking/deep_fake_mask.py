""" Masks functions for faceswap.py
Adapted from https://github.com/AlgoHunt/Face-Xray/blob/master/DeepFakeMask.py 
"""
import cv2
import numpy as np

from .masking import Masking
from .utils import mask_from_points


class DeepFakeMasking(Masking):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parts = self.construct_parts()
        self.paths = [np.concatenate(part) for part in self.parts]

    @property
    def total(self) -> int:
        return len(self.paths)

    def construct_parts(self):
        raise NotImplementedError

    def transform_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        return landmarks

    def compute_mask(self, img: np.ndarray, landmarks, idx: int) -> np.ndarray:
        # h, w = img.shape[:2]
        path = self.paths[idx]
        new_landmarks = self.transform_landmarks(landmarks)
        points = new_landmarks[path]
        mask = mask_from_points(img, points)
        return mask


class DFLMasking(DeepFakeMasking):
    """DFL facial mask"""

    def construct_parts(self):
        jaw = (
            range(0, 17),
            range(48, 68),
            range(0, 1),
            range(8, 9),
            range(16, 17),
        )
        eyes = (
            range(17, 27),
            range(0, 1),
            range(27, 28),
            range(16, 17),
            range(33, 34),
        )
        nose_ridge = (range(27, 31), range(33, 34))
        parts = [jaw, eyes, nose_ridge]
        return parts


class ComponentsMasking(DeepFakeMasking):
    """Component model mask"""

    def construct_parts(self):
        r_jaw = (range(0, 9), range(17, 18))
        l_jaw = (range(8, 17), range(26, 27))
        r_cheek = (range(17, 20), range(8, 9))
        l_cheek = (range(24, 27), range(8, 9))
        nose_ridge = (
            range(19, 25),
            range(8, 9),
        )
        r_eye = (
            range(17, 22),
            range(27, 28),
            range(31, 36),
            range(8, 9),
        )
        l_eye = (
            range(22, 27),
            range(27, 28),
            range(31, 36),
            range(8, 9),
        )
        nose = (range(27, 31), range(31, 36))
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]
        return parts


class ExtendedMasking(ComponentsMasking):
    """Extended mask
    Based on components mask. Attempts to extend the eyebrow points up the forehead
    """

    def transform_landmarks(self, landmarks):
        """Transform landmarks to extend the eyebrow points up the forehead"""
        new_landmarks = landmarks.copy()
        # mid points between the side of face and eye point
        ml_pnt = (new_landmarks[36] + new_landmarks[0]) // 2
        mr_pnt = (new_landmarks[16] + new_landmarks[45]) // 2

        # mid points between the mid points and eye
        ql_pnt = (new_landmarks[36] + ml_pnt) // 2
        qr_pnt = (new_landmarks[45] + mr_pnt) // 2

        # Top of the eye arrays
        bot_l = np.array(
            (
                ql_pnt,
                new_landmarks[36],
                new_landmarks[37],
                new_landmarks[38],
                new_landmarks[39],
            )
        )
        bot_r = np.array(
            (
                new_landmarks[42],
                new_landmarks[43],
                new_landmarks[44],
                new_landmarks[45],
                qr_pnt,
            )
        )

        # Eyebrow arrays
        top_l = new_landmarks[17:22]
        top_r = new_landmarks[22:27]

        # Adjust eyebrow arrays
        new_landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        new_landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

        return new_landmarks
