from math import ceil, floor

import cv2
import numpy as np

from .masking import Masking
from .utils import mask_from_points
from src.utils.bbox import landmarks_to_bbox


class GridMasking(Masking):
    def __init__(self, rows: int = 4, cols: int = 4, **kwargs):
        super(GridMasking, self).__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    @property
    def total(self) -> int:
        return self.rows * self.cols

    # def mask_to_points(self, mask):
    #    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #    for cnt in cnts:
    #        cv2.polylines(out, [cnt], True, 255, 1, lineType=8)

    def compute_mask(self, img: np.ndarray, landmarks, idx: int) -> np.ndarray:
        h, w = img.shape[:2]
        landmarks = landmarks[:68]
        # if idx is None:
        #    idx = np.random.randint(0, self.total)
        r, c = divmod(idx, self.cols)

        # pixel related
        xmin, ymin, xmax, ymax = landmarks_to_bbox(landmarks)
        dx = ceil((xmax - xmin) / self.cols)
        dy = ceil((ymax - ymin) / self.rows)

        mask = np.zeros((h, w), dtype=np.uint8)

        # fill the cell mask
        x0, y0 = floor(xmin + dx * c), floor(ymin + dy * r)
        x1, y1 = floor(x0 + dx), floor(y0 + dy)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

        # merge the cell mask with the convex hull
        ch = mask_from_points(img, landmarks)
        # ch = cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)
        # mask = (mask & ch) / 255.0
        mask = cv2.bitwise_and(mask, mask, mask=ch)
        # cv2.bitwise_or(img, d_3c_i)

        return mask
