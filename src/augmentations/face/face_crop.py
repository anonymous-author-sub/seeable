from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.utils.bbox import bbox_width_height, landmarks_to_bbox

from .face_transform import FaceTransform


def crop_face(
    img: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    bbox: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
    margin=False,
    crop_by_bbox=True,
    phase="train",
):
    assert phase in ["train", "val", "test"]
    assert any(x is not None for x in (landmarks, bbox))
    H, W = img.shape[:2]

    # process base bounding box
    bb = bbox if crop_by_bbox else landmarks_to_bbox(landmarks)
    w, h = bbox_width_height(bb)

    # left, top, right, bottom
    # x0_new, y0_new, x1_new, y1_new
    # margin coef 0
    if crop_by_bbox:
        margin_c0 = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        margin_c0 = np.array([0.125, 0.5, 0.125, 0.2])
    # margin coef 1
    if margin:
        margin_c1 = np.array([4, 2, 4, 2])
    elif phase == "train":
        margin_c1 = 0.2 + np.random.rand(4) * 0.6
    else:
        margin_c1 = np.array([0.5, 0.5, 0.5, 0.5])
    base_margin = np.array([w, h, w, h], dtype=np.float32)
    margin = base_margin * margin_c0 * margin_c1

    # w0, h0, w1, h1 = margin
    # x0_new = max(0, int(x0 - w0))
    # y0_new = max(0, int(y0 - h0))
    # y1_new = min(H - 1, int(y1 + h1))
    # x1_new = min(W - 1, int(x1 + w1))
    cropped_bbox = (bb + (np.array([-1, -1, +1, +1]) * margin)).astype(np.int32)
    cropped_bbox = np.clip(cropped_bbox, 0, [W - 1, H - 1] * 2)
    x0_new, y0_new, x1_new, y1_new = cropped_bbox

    # extract the face
    cropped_image = img[y0_new : y1_new + 1, x0_new : x1_new + 1]

    # crop image
    out = dict()
    out["cropped_bbox"] = cropped_bbox
    out["cropped_image"] = cropped_image

    # translate landmarks and bbox to the new coordinate system with origin at the
    # top left corner of the cropped image
    out["local"] = dict()
    if landmarks is not None:
        landmarks_cropped = np.zeros_like(landmarks)
        landmarks_cropped[:, 0] = landmarks[:, 0] - x0_new
        landmarks_cropped[:, 1] = landmarks[:, 1] - y0_new
        out["local"]["landmarks"] = landmarks_cropped.astype(np.int32)
    if bbox is not None:
        bbox_cropped = bbox - np.array([x0_new, y0_new] * 2)
        out["local"]["bbox"] = bbox_cropped.astype(np.int32)
    if keypoints is not None:
        keypoints_cropped = np.zeros_like(keypoints)
        keypoints_cropped[:, 0] = keypoints[:, 0] - x0_new
        keypoints_cropped[:, 1] = keypoints[:, 1] - y0_new
        out["local"]["keypoints"] = keypoints_cropped.astype(np.int32)

    # return
    return out


class FaceCrop(FaceTransform):
    def __init__(
        self,
        use_margin: bool = False,
        crop_by_bbox: bool = True,
        phase: str = "train",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.use_margin = use_margin
        self.crop_by_bbox = crop_by_bbox
        self.phase = phase
        self.is_train = phase == "train"

        # compute margin np.array([left, top, right, bottom])
        # relative to the width and height of the bounding box
        # depends on (crop_by_bbox, use_margin, is_train)
        self.rel_margin = self.get_rel_margin(crop_by_bbox, use_margin, self.is_train)

    @staticmethod
    def get_rel_margin(
        crop_by_bbox: bool, use_margin: bool, is_train: bool
    ) -> np.ndarray:
        """
        left, top, right, bottom
        x0_new, y0_new, x1_new, y1_new
        """
        # margin coef 0
        if crop_by_bbox:
            margin_c0 = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            margin_c0 = np.array([0.125, 0.5, 0.125, 0.2])

        # margin coef 1
        if use_margin:
            margin_c1 = np.array([4, 2, 4, 2])
        elif is_train:
            margin_c1 = 0.2 + np.random.rand(4) * 0.6
        else:
            margin_c1 = np.array([0.5, 0.5, 0.5, 0.5])

        rel_margin = margin_c0 * margin_c1
        return rel_margin

    @staticmethod
    def compute_bbox_margin(
        bbox_base: np.ndarray, rel_margin: np.ndarray
    ) -> np.ndarray:
        """Compute margin (in px) from the base bbox and the relative margin"""
        assert bbox_base.shape == (4,)
        w, h = bbox_width_height(bbox_base)
        base_margin = np.array([w, h, w, h], dtype=np.float32)
        margin = base_margin * rel_margin
        return margin

    @staticmethod
    def compute_bbox_crop(bbox_base, margin, width: int, height: int) -> np.ndarray:
        """Compute the crop bbox from the base bbox and the margin"""
        bbox_crop = bbox_base + (np.array([-1, -1, +1, +1]) * margin)
        bbox_crop = bbox_crop.astype(np.int32)
        bbox_crop = np.clip(bbox_crop, 0, [width - 1, height - 1] * 2)
        # x0_new, y0_new, x1_new, y1_new = bbox_crop
        return bbox_crop

    @staticmethod
    def get_crop_meta(
        face: Dict,
        rel_margin: np.ndarray,
        crop_by_bbox: bool,
        width: int,
        height: int,
    ) -> Dict[str, np.ndarray]:
        landmarks = face.get("landmarks")
        bbox_landmarks = face.get("bbox_landmarks")
        bbox = face.get("bbox")

        # get bbox_base: {bbox(landmarks), bbox}
        if crop_by_bbox and bbox is not None:
            bbox_base = bbox
        elif bbox_landmarks is not None:
            bbox_base = bbox_landmarks
        elif landmarks is not None:
            bbox_base = landmarks_to_bbox(landmarks)
        else:
            raise ValueError("No bbox_crop")
        # compute margin in px
        margin = FaceCrop.compute_bbox_margin(bbox_base, rel_margin)
        # compute effictive crop bbox
        bbox_crop = FaceCrop.compute_bbox_crop(bbox_base, margin, width, height)
        return {
            "rel_margin": rel_margin,
            "margin": margin,
            "bbox_base": bbox_base,
            "bbox_crop": bbox_crop,
        }

    # -------------------------------------------------------------------------

    def get_transform_init_args_names(self):
        return ("use_margin", "crop_by_bbox", "phase")

    def setup(self, **kwargs) -> dict:
        face = kwargs["face"]
        height, width = kwargs["image"].shape[:2]
        aug_meta = self.get_crop_meta(
            face,
            self.rel_margin,
            self.crop_by_bbox,
            width,
            height,
        )
        augs = kwargs[self.AUGS]
        augs[self.__class__.__name__].update(aug_meta)
        return augs

    # -------------------------------------------------------------------------

    @staticmethod
    def _apply_to_image(img, aug=None, **params) -> np.ndarray:
        bbox_crop = aug["bbox_crop"]
        x0, y0, x1, y1 = bbox_crop
        img_crop = img[y0 : y1 + 1, x0 : x1 + 1, :]
        return img_crop

    @staticmethod
    def _apply_to_landmarks(landmarks, aug=None, **params) -> np.ndarray:
        bbox_crop = aug["bbox_crop"]
        x0, y0, x1, y1 = bbox_crop
        landmarks_crop = np.zeros_like(landmarks, dtype=np.int32)
        landmarks_crop[:, 0] = landmarks[:, 0] - x0
        landmarks_crop[:, 1] = landmarks[:, 1] - y0
        return landmarks_crop

    @staticmethod
    def _apply_to_bbox(bbox, aug, **params) -> np.ndarray:
        """bbox format x0, y0, x1, y1"""
        assert bbox.shape == (4,)
        bbox_crop = aug["bbox_crop"]
        x0, y0, x1, y1 = bbox_crop
        new_bbox = bbox - np.array([x0, y0, x0, y0])
        # new_bbox = new_bbox.astype(np.int32)
        return new_bbox
