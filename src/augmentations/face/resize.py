from typing import Callable, Dict, List, Optional, Tuple

import albumentations.augmentations.geometric.functional as F
import cv2
import numpy as np
from loguru import logger

from .face_transform import FaceTransform


class FaceResize(FaceTransform):
    """Resize the input to the given height and width.
    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        height: int,
        width: int,
        interpolation=cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(FaceResize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    # --------------------------------------------------------------------------

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")

    def setup(self, **kwargs) -> dict:
        height, width = kwargs["image"].shape[:2]
        scale_x = self.width / width
        scale_y = self.height / height
        aug = {
            "interpolation": self.interpolation,
            "width": self.width,
            "height": self.height,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "original_width": width,
            "original_height": height,
        }
        augs = kwargs[self.AUGS]
        augs[self.__class__.__name__].update(aug)
        return augs

    # --------------------------------------------------------------------------

    @staticmethod
    def _apply_to_image(img, aug=None, **params) -> np.ndarray:
        return F.resize(
            img,
            height=aug["height"],
            width=aug["width"],
            interpolation=aug["interpolation"],
        )

    @staticmethod
    def _apply_to_landmarks(landmarks, aug=None, **params) -> np.ndarray:
        # return F.keypoint_scale(landmarks, scale_x, scale_y)
        scale_x, scale_y = aug["scale_x"], aug["scale_y"]
        scale = np.array([scale_x, scale_y], dtype=np.float32)
        new_landmarks = landmarks * scale
        return new_landmarks

    @staticmethod
    def _apply_to_bbox(bbox, aug=None, **params) -> np.ndarray:
        """bbox format x0, y0, x1, y1"""
        # return F.keypoint_scale(keypoint, scale_x, scale_y)
        scale_x, scale_y = aug["scale_x"], aug["scale_y"]
        scale = np.array([scale_x, scale_y, scale_x, scale_y])
        new_bbox = bbox.astype(np.float32) * scale
        return new_bbox
