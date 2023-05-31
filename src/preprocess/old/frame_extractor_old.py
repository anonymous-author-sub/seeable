import os
import cv2
import json
import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Optional

import torch
import torch.backends.cudnn as cudnn

from hydra.utils import instantiate

#  from facenet_pytorch import MTCNN
# from models.retina import Retina
from .utils import get_boundingbox, get_frame_types, select_samples_index
from src.model.detector import Detector


class FrameExtractor:
    def __init__(
        self,
        cfg,
        dataset,
        picture_type: str="I",
        root: str = None,
        detector: Optional[Detector] = None,
        extension: str = ".png",
    ):
        self.cfg = cfg
        self.dataset = dataset  # instantiate(dataset)
        # configure the frames to be extracted
        if not isinstance(picture_type, (str, int)):
            raise Exception("[FrameExtractor] picture_type must be a string or an int")
        if isinstance(picture_type, str):
            if picture_type.isnumeric():
                picture_type = int(picture_type)
                if picture_type <= 0:
                    raise Exception("[FrameExtractor] picture_type must be a positive int")
            elif picture_type in ["I", "B", "P"] + ["all"]:
                picture_type = picture_type
            else:
                raise Exception("[FrameExtractor] picture_type : int, I, B, P, or all")
        self.picture_type = picture_type
        self.extension = extension

        # configure
        self.path = Path(root) if root is not None else Path(cfg.dataset.root)
        print("[FrameExtractor] self.path", self.path)

        # create subdir and files
        self.path_images = self.path / "images"
        self.path_metas = self.path / "metas"

        exit()

    def process_frame(self, frame, width: int, height: int) -> Tuple[np.ndarray, dict]:
        out = self.detector(frame)
        if out is None:
            raise Exception("[FrameExtractor] face not detected")
        boxes, probs, points = out

        # get largest face
        maxi = np.argmax(boxes[:, 2] - boxes[:, 0])
        box, landms = boxes[maxi], points[maxi]

        # crop the face
        x, y, size = get_boundingbox(boxes.flatten(), width, height)
        face = frame[y : y + size, x : x + size]

        meta = {}
        meta["box"] = box.tolist()
        meta["landms"] = landms.ravel().tolist()

        return face, meta

    def process_video(self, filepath: str, prefix: str = "") -> dict:
        v_cap = cv2.VideoCapture(filepath)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        samples, types = select_samples_index(
            filepath, self.picture_type, v_len, return_types=True
        )

        metas, info = {}, []
        for frame_idx, frame_type in zip(samples, types):
            v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = v_cap.read()
            height, width = frame.shape[:2]
            frame_name = Path(f"{prefix}{frame_idx:06d}{self.extension}")
            info.append((frame_name.name, frame_idx, frame_type))

            # save full / raw frame
            if True:
                p = self.path_full / frame_name
                p.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(p), frame)

            # detect, crop and save face
            if self.detector is not None:
                try:
                    face, meta = self.process_frame(frame, width, height)
                except Exception as e:  # face not detected
                    print(e)
                    p = self.path_faces_no / frame_name
                    p.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(p), frame)
                else:  # face detected
                    p = self.path_faces / frame_name
                    p.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(p), face)
                    metas[str(frame_name)] = meta

        return metas, info

    def run(self):
        # loop trough each split
        for i, name, in enumerate(self.dataset.names):
            # target : 0/1 (real/fake)
            # path   : dir path
            target = self.dataset.targets[i], 
            path = self.dataset.paths[i]
            #print(name, target, path)

            # for each dir, loop trough each video
            for file_path in path.iterdir():
                if not file_path.is_file():
                    print(f"{file_path} is not a file !")
                    continue
                
                prefix = f"{name}/{target}_{file_path.stem}_"
                file_metas, file_infos = self.process_video(str(file_path), prefix)



