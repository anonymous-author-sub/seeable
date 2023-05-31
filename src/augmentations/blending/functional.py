from typing import Tuple

import albumentations as A
import numpy as np


def bbox2(img) -> Tuple[int, int, int, int]:
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def randaffine(img, mask) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]

    # compute mask effective width height ratio
    print("randaffine", mask.shape)
    rmin, rmax, cmin, cmax = bbox2(mask)
    mw, mh = cmax - cmin, rmax - rmin
    # print("randaffine", mw, mh)
    dw, dh = mw / w, mh / h
    # print("randaffine", dw, dh)

    # affine transformation
    # translate_x = 3 * dw / 100
    # translate_y = 1.5 * dh / 100
    # scale_ratio = 5 * ((dw + dh) / 2) / 100

    translate_x = 3 / 100
    translate_y = 1.5 / 100
    scale_ratio = 2 / 100

    f = A.Affine(
        translate_percent={
            "x": (-translate_x, translate_x),
            "y": (-translate_y, translate_y),
        },
        scale=[1 - scale_ratio, 1 + scale_ratio],
        fit_output=False,
        p=1,
    )
    if 1:
        # print("randaffine", translate_x, translate_y, scale_ratio)
        transformed = f(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

    # elastic deformation
    if 1:
        g = A.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )
        mask = g(image=img, mask=mask)["mask"]

    return img, mask
