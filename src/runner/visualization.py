import random
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, Tuple, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.augmentations.face.face_crop import crop_face
from src.utils.bbox import bbox_size, landmarks_to_bbox, pairwise_iou
from src.utils.seed import seed_everything


def get_landmarks_and_bbox(frame: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return landmarks and bbox from frame by computing the max iou between
    - the largest retina detected face bbox
    - all the landmarks' bbox
    """
    landmarks_list = [np.array(face["points"]) for face in frame["dlib68"]]
    # bboxes_ld = [landmarks_to_bbox(ld) for ld in landmarks_list]
    bboxes_ld = [np.array(face["bbox"]) for face in frame["dlib68"]]
    sizes_ld = [bbox_size(b) for b in bboxes_ld]
    # keep the largest face detected
    i_ld = np.argmax(sizes_ld)
    landmarks, bbox_ld = landmarks_list[i_ld], bboxes_ld[i_ld]

    # retina
    bboxes = [np.array(face["bbox"]) for face in frame["retina"]]
    bboxes_iou = [pairwise_iou(bbox_ld, b) for b in bboxes]
    # keep the face with the highest iou with the largest face detected
    i_bb = np.argmax(bboxes_iou)
    bbox = bboxes[i_bb]

    return landmarks, bbox


def resize_face(face_dict: Dict, size: Tuple[int, int], original_size: Tuple[int, int]):
    # H, W = original_size
    H, W = face_dict["cropped_image"].shape[:2]
    h, w = size
    scales = np.array([w / W, h / H] * 2, dtype=np.float32)

    img = face_dict["cropped_image"]
    ld = face_dict["local"]["landmarks"]
    bbox = face_dict["local"]["bbox"]

    return {
        "cropped_image": cv2.resize(img, size, interpolation=cv2.INTER_AREA),
        "cropped_bbox": face_dict["cropped_bbox"],
        "local": {
            "landmarks": (ld * scales[:2]).astype(np.int32),
            "bbox": (bbox * scales).astype(np.int32),
        },
    }


class Visualization:
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset

        # cv2.namedWindow("Remote", cv2.WINDOW_AUTOSIZE)

    def visu(
        self,
        img: np.ndarray,
        landmarks: np.ndarray,
        bbox: np.ndarray,
        keypoints=None,
        title: Optional[str] = None,
    ):
        out = img.copy()

        # draw landmarks
        for x, y in landmarks:
            x, y = int(x), int(y)
            out = cv2.circle(out, (x, y), 1, (0, 0, 255), -1)

        # draw keypoints as losange
        if keypoints is not None:
            for x, y in keypoints:
                x, y = int(x), int(y)
                out = cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
                out = cv2.line(out, (x - 2, y), (x + 2, y), (0, 255, 0), 1)
                out = cv2.line(out, (x, y - 2), (x, y + 2), (0, 255, 0), 1)

        # draw bbox landmarks
        bbox_ld = landmarks_to_bbox(landmarks)
        x0_ld, y0_ld, x1_ld, y1_ld = list(map(int, bbox_ld))
        out = cv2.rectangle(out, (x0_ld, y0_ld), (x1_ld, y1_ld), (0, 0, 255), 1)

        # draw bbox
        x0, y0, x1, y1 = list(map(int, bbox))
        out = cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 1)

        # display
        if 0:
            face_img = cv2.resize(face_img, (256, 256), interpolation=cv2.INTER_LINEAR)

        if title is None:
            title = "face"
        cv2.imshow(title, out)

        ch = cv2.waitKey(0)
        if ch == ord(" "):
            pass
        if ch == 27:
            return 1

    def show_frame(self, frame: Dict):
        path = frame["path"]
        img = cv2.imread(frame["path"])
        assert img is not None, f"Image not found: {frame['path']}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if any([not frame.get(x) for x in ["retina", "dlib68"]]):
            print("No face detected:", path)
            return
        landmarks, bbox = get_landmarks_and_bbox(frame)

        # extract the face
        face_dict = crop_face(img, landmarks, bbox, margin=True, crop_by_bbox=False)

    def run_frame(self, sample):
        for frame in sample["frames"]:
            r = self.show_frame(frame)
            # print({k: v for k, v in frame.items() if k != "image"})
            if r == 1:
                return

    def run_face(self, sample):
        video, frame, face = sample["video"], sample["frame"], sample["face"]
        # "bbox": stats["bbox_face"],
        # "bbox_landmarks": stats["bbox_land"],
        # "landmarks": landmarks,
        # "iou": iou,
        if face is None:
            return

        img = sample["image"]
        # path = sample["frames"][face["index"]]["path"]
        # img = cv2.imread(path)
        # assert img is not None, f"Image not found: {sample['path']}"
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cropped = crop_face(
            img,
            face["landmarks"],
            face["bbox"],
            face["keypoints"],
            margin=True,
            crop_by_bbox=False,
        )

        return self.visu(
            cropped["cropped_image"],
            cropped["local"]["landmarks"],
            cropped["local"]["bbox"],
            keypoints=cropped["local"]["keypoints"],
            title=f"{video['index']}-{frame['index']}",
        )

    # SEE : https://www.programcreek.com/python/example/67893/cv2.namedWindow

    def run(self) -> None:
        seed_everything(42)
        I = list(range(len(self.dataset)))
        # random.shuffle(I)
        for idx in I:
            sample = self.dataset[idx]
            if self.dataset.level == "frame":
                self.run_frame(sample)
            elif self.dataset.level == "face":
                self.run_face(sample)
        cv2.destroyAllWindows()
