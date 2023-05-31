import random
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

# import BoxInternalType
import albumentations.augmentations.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.augmentations import masking
from src.utils.types import BboxArray, LandmarksArray

from .face_transform import FaceTransform


def get_palette(n: int) -> np.ndarray:
    cmap = plt.cm.get_cmap(name="hsv", lut=n + 1)
    colors = np.array([cmap(i)[:3] for i in range(n)])
    return (255 * colors).astype(int).tolist()


class FaceMask(FaceTransform):
    """Flip the input horizontally around the x-axis.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, p=0.5, always_apply=False):
        super(FaceMask, self).__init__(p, always_apply)

        # self.masking = MeshgridMasking()
        # self.masking = SladdMasking()
        # self.masking = GridMasking()

        self.masking_schemes = {
            "grid": dict(rows=4, cols=4),
            "sladd": dict(compose=False),
            "meshgrid": dict(),
            "whole": dict(),
            "dfl": dict(),
            "components": dict(),
            "extended": dict(),
        }

        # self.masking = masking.from_name("grid", rows=2, cols=3)
        # self.masking = masking.from_name("sladd", compose=False)
        # self.masking = masking.from_name("meshgrid")
        # self.masking = masking.from_name("whole")
        # self.masking = masking.from_name("dfl")
        # self.masking = masking.from_name("components")
        # self.masking = masking.from_name("extended")

    def setup(self, **kwargs) -> dict:

        img = kwargs["image"]
        height, width = img.shape[:2]
        face = kwargs["face"]
        landmarks = face["landmarks"]

        masking_scheme = random.choice(list(self.masking_schemes.keys()))
        submasks = masking.from_name(
            masking_scheme, **self.masking_schemes[masking_scheme]
        )

        n = submasks.total
        colors = get_palette(n)
        aug = {"masks": []}
        for i, color in enumerate(colors):
            mask = submasks(img, landmarks, i)
            aug["masks"].append(
                {
                    "mask": mask,
                    "color": color,
                    "index": i,
                }
            )

        augs = kwargs[FaceTransform.AUGS]
        augs[self.__class__.__name__].update(aug)
        return augs

    @staticmethod
    def _apply_to_image(img, aug=None, **params) -> np.ndarray:
        height, width = img.shape[:2]
        out = img.copy()

        # for mask in aug["masks"]:
        #    img = cv2.bitwise_and(img, mask["mask_binary"])

        # contours
        outline = np.zeros((height, width), dtype=np.uint8)
        # intersection
        acc = np.zeros((height, width), dtype=np.uint8)

        masks = aug["masks"]
        # masks = random.sample(masks, k=1)

        for mask_data in masks:
            mask = mask_data["mask"]
            color = mask_data["color"]
            # fill only the outline of the mask
            # outline = cv2.polylines(outline, [mask["points"]], True, 255, 1, lineType=8)

            # detect contour of the mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(outline, contours, -1, 255, 1, lineType=8)

            acc[mask > 0] += 1
            out[mask > 0] = color

        out[acc > 1] = 0
        out[outline > 0] = 255 - out[outline > 0]  # 255

        return out

    @staticmethod
    def _apply_to_bbox(bbox: BboxArray, **params) -> np.ndarray:
        return bbox

    @staticmethod
    def _apply_to_landmarks(landmarks: LandmarksArray, **params) -> np.ndarray:
        return landmarks
