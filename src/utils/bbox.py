from typing import Dict, Tuple

import numpy as np


def bbox_width_height(bbox: np.ndarray):
    x0, y0, x1, y1 = bbox
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    return width, height


def bbox_size(bbox: np.ndarray) -> int:
    width, height = bbox_width_height(bbox)
    return width * height


def landmarks_to_bbox(landmarks: np.ndarray) -> np.ndarray:
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    assert landmarks.shape[1] == 2
    x0, y0 = np.min(landmarks, axis=0)
    x1, y1 = np.max(landmarks, axis=0)
    bbox = np.array([x0, y0, x1, y1])
    return bbox


def pairwise_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# Numpy implementation for multiple bboxes


def bboxes_area(bboxes: np.ndarray) -> int:
    assert isinstance(bboxes, np.ndarray)
    assert bboxes.shape[1] == 4 and bboxes.shape[0] > 0
    X0, Y0, X1, Y1 = bboxes.T
    return (X1 - X0) * (Y1 - Y0)


def bboxes_intersection_over_union(
    boxes1: np.ndarray, boxes2: np.ndarray
) -> Dict[str, np.ndarray]:
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    assert isinstance(boxes1, np.ndarray) and isinstance(boxes2, np.ndarray)
    if len(boxes1.shape) == 1:
        boxes1 = boxes1[None, :]
    if len(boxes2.shape) == 1:
        boxes2 = boxes2[None, :]
    assert boxes1.shape[1] == 4 and boxes2.shape[1] == 4, (
        "box format is supposed to be: "
        + "top_left_x, top_left_y, bottom_right_x, bottom_right_y "
        + f"got {boxes1.shape} and {boxes2.shape}"
    )
    # degenerate boxes gives inf / nan results
    assert (boxes1 >= 0).all(), f"boxes1 can not contain negative values. {boxes1}"
    assert (boxes2 >= 0).all(), f"boxes2 can not contain negative values. {boxes2}"

    area1 = bboxes_area(boxes1)
    area2 = bboxes_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    # union = (area1 + area2 - inter)
    union = area1[:, None] + area2 - inter
    ious = inter / union

    return {
        "iou_matrix": inter / (area1[:, None] + area2 - inter),
        "inter_matrix": inter,
        "union_matrix": union,
        "area1": area1,
        "area2": area2,
    }
