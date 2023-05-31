import math
import numpy as np
from typing import Any


class BboxArray(np.ndarray):
    def __init__(self, *args, **kwargs):
        super(BboxArray, self).__init__(*args, **kwargs)


class LandmarksArray(np.ndarray):
    def __init__(self, *args, **kwargs):
        super(LandmarksArray, self).__init__(*args, **kwargs)


def check_type(x: Any) -> bool:
    """check if x is non degerate"""
    if isinstance(x, float):
        if math.isnan(x):
            return False
    elif isinstance(x, str):
        return True
    elif np.issubdtype(type(x), np.number):
        if np.isnan(x):
            return False
    return True


def convert_to_numpy_2D(data: Any, dtype=None) -> np.ndarray:
    """convert data to numpy array of len(shape)=2"""
    data = np.array(data)
    if data.ndim < 2:
        data = np.expand_dims(data, axis=0)
    if dtype is not None:
        data = data.astype(dtype)
    return data
