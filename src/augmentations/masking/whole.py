import cv2
import numpy as np

from .masking import Masking
from .utils import mask_from_points


class WholeMasking(Masking):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def total(self) -> int:
        return 1

    def compute_mask(self, img: np.ndarray, landmarks, idx: int) -> np.ndarray:
        h, w = img.shape[:2]
        mask = mask_from_points(img, landmarks)
        return mask
