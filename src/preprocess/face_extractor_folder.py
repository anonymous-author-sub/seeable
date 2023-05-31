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


# 12/12 son; femme le 15
# antoine le 15
# visite hebdomaaire (2 fois par semaine)
# rentrer ce WE, peut etre revenir à l'hopital


def round_floats(o):
    if isinstance(o, float):
        return round(o, 4) if o < 1 else round(o, 2)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


class FaceExtractorFolder:
    def __init__(
        self,
        cfg,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        detector: Optional[Detector] = None,
    ):
        self.cfg = cfg
        self.detector = detector

        # paths
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

        # self.detector_path = self.output_path / self.detector.name
        # self.detector_path.mkdir(exist_ok=True, parents=True)

    def extract_faces(self, frame: np.ndarray) -> dict:
        boxes, probs, points = self.detector(frame)
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

        images = [x for x in self.input_path.iterdir() if x.is_file()]
        images = sorted(images, key=lambda x: int(x.stem), reverse=False)

        tbar = tqdm(images, desc="images")
        for image in tbar:
            filename = image.stem
            out_path = self.output_path / f"{filename}.json"

            # skip if already exists
            if out_path.is_file():
                continue

            # extract faces
            frame: np.ndarray = load_rgb(str(image))
            data: list = self.extract_faces(frame)
            if not data:
                print("[FaceExtractorFolder] no faces detected", filename)

            # save the json file
            data_json = json.dumps(round_floats(data))
            with open(str(out_path), "w") as f:
                f.write(data_json)

            total += 1
