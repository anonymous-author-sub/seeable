from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from src.augmentations.base_transform import BaseTransform
from src.utils.types import BboxArray, LandmarksArray


class FaceTransform(BaseTransform):
    """Transform for deepfakes task:
    targets: image, face

    functional
        - image: apply to image
        - face: apply to landmarks and bbox by detecting the type of the value
    """

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "face": self.apply_face,
        }

    @property
    def target_dependence(self) -> Dict:
        return {}

    # -------------------------------------------------------------------------

    @staticmethod
    def _apply_to_image(img, **params) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _apply_to_landmarks(landmarks, **params) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _apply_to_bbox(bbox, **params) -> np.ndarray:
        raise NotImplementedError

    # -------------------------------------------------------------------------

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img_out = self._apply_to_image(img, **params)
        return img_out

    def apply_face(self, face: np.ndarray, **params) -> dict:
        for k, v in face.items():
            if isinstance(v, LandmarksArray):
                out_v = self._apply_to_landmarks(v, **params)
                face[k] = out_v.view(LandmarksArray)
            elif isinstance(v, BboxArray):
                out_v = self._apply_to_bbox(v, **params)
                face[k] = out_v.view(BboxArray)
        return face
