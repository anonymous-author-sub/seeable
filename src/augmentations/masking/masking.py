from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.types import LandmarksArray


class Masking(ABC):
    """
    Base class for sub-masking scheme
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def total(self) -> int:
        ...

    @abstractmethod
    def compute_mask(img: np.ndarray, landmarks, idx: int) -> np.ndarray:
        ...

    def __call__(
        self, img: np.ndarray, landmarks: LandmarksArray, idx: Optional[int] = None
    ) -> np.ndarray:
        """return a mask of the same size as img with
        1s in the masked region and 0s elsewhere
        If idx is None, a random index is chosen
        """
        if idx is None:
            idx = np.random.randint(0, self.total)
        assert isinstance(idx, int) and 0 <= idx < self.total
        return self.compute_mask(img, landmarks, idx)
