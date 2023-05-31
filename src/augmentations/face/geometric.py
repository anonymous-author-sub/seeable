from typing import Callable, Dict, Optional, Sequence, Tuple, Union

# import BoxInternalType
import albumentations.augmentations.functional as F

# DualTransform
import numpy as np
from loguru import logger

from src.utils.landmarks import get_symmetric_permutation
from src.utils.types import BboxArray, LandmarksArray

from .face_transform import FaceTransform

BoxType = Tuple[float, float, float, float]
KeypointType = Tuple[float, float, float, float]


class FaceHorizontalFlip(FaceTransform):
    """Flip the input horizontally around the x-axis.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    @staticmethod
    def _apply_to_image(img, **params) -> np.ndarray:
        return F.hflip(img)

    @staticmethod
    def _apply_to_bbox(bbox: BboxArray, **params) -> np.ndarray:
        width, height = params["width"], params["height"]
        # logger.info(str(params))
        # H, W = params.get("image").shape[:2]
        # return F.bbox_hflip(bbox, height, width)
        x0, y0, x1, y1 = bbox
        x0_new = width - 1 - x0
        x1_new = width - 1 - x1
        new_bbox = np.array([x0_new, y0, x1_new, y1], dtype=np.int32)
        return new_bbox

    @staticmethod
    def _apply_to_landmarks(landmarks: LandmarksArray, **params) -> np.ndarray:
        width, height = params["width"], params["height"]
        n_ld = landmarks.shape[0]
        v_perm: np.array = get_symmetric_permutation(n_ld)
        # landmark_new = np.zeros_like(landmark)
        landmark_new: np.array = landmarks[v_perm]
        # X axis flip # ERROR FIX W -> W-1
        landmark_new[:, 0] = (width - 1) - landmark_new[:, 0]
        return landmark_new
