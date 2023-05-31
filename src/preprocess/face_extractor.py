import os
import json
import cv2
import torch
import pandas as pd
import numpy as np

# from PIL import Image
from tqdm import tqdm
from typing import Tuple, Optional
from pathlib import Path

from src.detector import Detector
from .utils import get_boundingbox, get_frame_types, select_samples_index


def load_rgb(file_path: str, size: Optional[int] = None) -> np.ndarray:
    try:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if size is not None:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return img
    except Exception as e:
        print(e)
        print("-" * 40)
        print(file_path)
        print("-" * 40)
        raise ValueError("load_rgb failed")


def round_floats(o):
    if isinstance(o, float):
        return round(o, 4) if o < 1 else round(o, 2)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


class FaceExtractor:
    def __init__(
        self,
        cfg,
        root: str = None,
        detector: Optional["Detector"] = None,
    ):
        self.cfg = cfg
        # self.root = root
        self.detector = detector
        # self.detector = self.detector.to('cuda:0')

        self.path = Path(root)
        print("[FrameExtractor] self.path", self.path)

        # create subdir and files
        self.path_images = self.path / "images"

        self.path_detector = self.path / self.detector.name
        self.path_detector.mkdir(exist_ok=True, parents=True)

    def extract_faces(self, frame: np.ndarray) -> dict:
        boxes, probs, points = self.detector(frame)
        if False:
            print(f"boxes: {boxes.shape}", boxes)
            print(f"probs: {probs.shape}", probs)
            print(f"points: {points.shape}", points)
        out = []

        detected = len(probs)
        for i in range(detected):
            try:
                item = {}
                item["bbox"] = boxes[i].tolist()
                item["prob"] = probs[i].tolist()
                item["points"] = points[i].tolist()
            except TypeError:
                pass
            else:
                out.append(item)
        return out

    def run(self):
        total = 0

        # folders = list(self.path_images.glob("*"))
        # keep only directories
        folders = [x for x in self.path_images.iterdir() if x.is_dir()]
        folders = sorted(folders, key=lambda x: int(x.name), reverse=False)

        for video_dir in tqdm(folders, desc="videos"):

            # create the dir for the video detector
            out_dir = self.path_detector / video_dir.name
            out_dir.mkdir(exist_ok=True, parents=True)

            frames = list(video_dir.glob("*"))
            frames = sorted(frames, key=lambda x: int(x.stem), reverse=False)

            for frame_path in frames:
                # print(frame_path.name)
                frame_idx = frame_path.stem
                out_path = out_dir / f"{frame_idx}.json"

                if not out_path.is_file():
                    # extract faces
                    frame = load_rgb(str(frame_path))
                    data = self.extract_faces(frame)
                    if not data:
                        print(
                            "[FaceExtractor] no faces detected", out_dir.name, frame_idx
                        )

                    # save the json file
                    data_json = json.dumps(round_floats(data))
                    with open(str(out_path), "w") as f:
                        f.write(data_json)

                total += 1
