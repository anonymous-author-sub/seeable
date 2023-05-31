from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np


import blend as B
from masking import GridMasking, MeshgridMasking, SladdMasking
from forgery import SyntheticForgery
from funcs import IoUfrom2bboxes, RandomDownScale, crop_face
from src.utils.landmarks import get_symmetric_permutation
from .functional import bbox2, randaffine


def resize_img(img: np.ndarray, size: int) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img_resized.astype("float32") / 255


def bbox2(img):
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def get_transforms():
    return A.Compose(
        [
            A.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=(-0.3, 0.3),
                sat_shift_limit=(-0.3, 0.3),
                val_shift_limit=(-0.3, 0.3),
                p=0.3,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3
            ),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
        ],
        additional_targets={f"image1": "image"},
        p=1.0,
    )


def get_source_transforms():
    return A.Compose(
        [
            A.Compose(
                [
                    A.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                    A.HueSaturationValue(
                        hue_shift_limit=(-0.3, 0.3),
                        sat_shift_limit=(-0.3, 0.3),
                        val_shift_limit=(-0.3, 0.3),
                        p=1,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.1),
                        contrast_limit=(-0.1, 0.1),
                        p=1,
                    ),
                ],
                p=1,
            ),
            A.OneOf(
                [
                    RandomDownScale(p=1),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ],
                p=1,
            ),
        ],
        p=1.0,
    )


class BlendingAugmentation:
    def __init__(self) -> None:
        super().__init__()

        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()

    def self_blending(self, img, mask):
        # (1/2) choose source-target for blending
        # (1/2) choose source-target for augmentation
        aug_source = bool(np.random.rand() < self.p_source_target)
        aug_target = not aug_source

        # source
        if aug_source:
            source = img.copy().astype(np.uint8)
            source = self.source_transforms(image=source)["image"]
        source, mask = randaffine(source, mask)

        # get target
        if aug_target:
            target = img.copy().astype(np.uint8)
            target = self.source_transforms(image=target)["image"]

        # blending
        fake, mask = B.dynamic_blend(source, target, mask)
        fake = fake.astype(np.uint8)

        return target, fake, mask
