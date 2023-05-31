import cv2
import numpy as np

from .masking import Masking
from .utils import landmarks_68_symmetries


class MeshgridMasking(Masking):
    areas = [
        [1, 2, 3, 4, 5, 6, 7, 49, 32, 40, 41, 42, 37, 18],
        [37, 38, 39, 40, 41, 42],  # left eye
        [18, 19, 20, 21, 22, 28, 40, 39, 38, 37],
        [28, 29, 30, 31, 32, 40],
    ]
    areas_asym = [
        [20, 21, 22, 28, 23, 24, 25],  # old [22, 23, 28],
        [31, 32, 33, 34, 35, 36],
        [32, 33, 34, 35, 36, 55, 54, 53, 52, 51, 50, 49],
        [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
        [7, 8, 9, 10, 11, 55, 56, 57, 58, 59, 60, 49],
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        sym, _ = landmarks_68_symmetries()
        # construct list of points paths
        paths = []
        paths += self.areas_asym  # asymmetrical areas
        paths += self.areas  # left
        paths += [[sym[ld68_id] for ld68_id in area] for area in self.areas]  # right
        assert len(paths) == self.total
        self.paths = paths

    @property
    def total(self) -> int:
        total = len(self.areas_asym) + len(self.areas) * 2
        return total

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

    def compute_mask(self, img: np.ndarray, landmarks, idx: int) -> np.ndarray:
        h, w = img.shape[:2]

        path = self.paths[idx]
        new_landmarks = self.transform_landmarks(landmarks)
        points = [new_landmarks[ld68_id - 1] for ld68_id in path]
        points = np.array(points, dtype=np.int32)

        # cv2.fillConvexPoly(out, points, 255, lineType=4)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)

        return mask
