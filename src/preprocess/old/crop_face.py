import os
import cv2
import json

from pathlib import Path
from PIL import Image
from tqdm import tqdm

import subprocess
import pandas as pd
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from facenet_pytorch import MTCNN

from models.retina import Retina
from .utils import get_boundingbox, get_frame_types, select_samples_index


class FaceGetter:
    def __init__(self, cfg):
        # self.mode = "train"
        self.save_face = True  # true: save faces; false: only frames with json metadata
        self.select_nums = "I"
        self.base_path = Path("/media/borutb/disk11/DataBase/DeepFake/Celeb-DF-v2/")

        # The base dir of DeepFake dataset
        self.video_root = self.base_path / "original_videos"
        # Where to save cropped training faces
        # self.output_path = None # in case you dont want to save anything
        self.output_path = self.base_path / self.mode
        # the given train-list.txt or test-list.txt file
        self.txt_path = self.base_path / f"{self.base_path.name}-{self.mode}-list.txt"

        self.dir_name = "faces" if self.save_face else f"{self.select_nums}-frames"

        # where to save frames meta data  - coords of faces
        self.meta_path = self.output_path / f"{self.dir_name}_meta.json"

        # init Face detectors
        self.device = "cuda:0"
        self.mtcnn = MTCNN(device=self.device).eval()
        self.retina = Retina(threshold=0.9, device=self.device).eval()
        # cudnn.benchmark = True

    def get_face(self, vframe, width: int, height: int, use_crop=True) -> dict:
        if self.method == "retina":
            boxes, probs, points = self.retina.detect_faces(vframe)
        elif self.method == "mtcnn":
            image = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            boxes, probs, points = self.mtcnn.detect(image, landmarks=True)
        else:
            raise Exception(f"method not supported : {self.method}")

        # get largest face
        maxi = np.argmax(boxes[:, 2] - boxes[:, 0])
        box, landms = boxes[maxi], points[maxi]

        face = {}
        if use_crop:
            x, y, size = get_boundingbox(boxes.flatten(), width, height)
            face["vframe_cropped"] = vframe[y : y + size, x : x + size]
        face["box"] = box.tolist()
        face["landms"] = landms.ravel().tolist()
        return face

    def get_faces(self, video_path, save_path, video_type):
        """
        :select_nums: num of frames: 0=all, 'I'=I-frames
        """
        v_cap = cv2.VideoCapture(video_path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        samples = select_samples_index(video_path, self.select_nums, v_len)

        use_crop = True

        metas_frames = {}
        for sample_idx in samples:
            v_cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
            _, vframe = v_cap.read()
            height, width = vframe.shape[:2]
            s = str(sample_idx).zfill(3)

            try:
                face = self.get_face(vframe, width, height, use_crop=use_crop)
            except:  # face not detected
                print(video_path)
                if save_path:
                    # TODO
                    subdir = video_path.split("/")[-2]
                    outdir = self.output_path / "nofaces"
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    filename = outdir + subdir / f"_{video_path.stem}_{s}_noface.png"
                    cv2.imwrite(filename, vframe)

            else:  # face detected
                if self.save_face:
                    filename = str(save_path / f"{s}.png")
                    image = face["vframe_cropped"]
                else:
                    key = f"{video_type}-{video_path.stem}_{s}"
                    metas_frames[key] = dict(box=face["box"], landms=face["landms"])
                    filename = str(save_path / f"{key}.png")
                    image = vframe
                if save_path:
                    cv2.imwrite(filename, image)

        v_cap.release()
        return v_len, metas_frames

    def run(self):
        data = []
        with open(self.txt_path, "r") as f:
            for line in f.readlines():
                video_type, video_name = line.split(" ")  # [0:1] [2:]
                data.append((video_type, Path(video_name)))

        metas, all_frames = {}, 0
        for video_type, video_name in tqdm(data):
            video_path = self.video_root / video_name

            save_path = None
            if self.output_path:
                save_path = self.video_root / f"{self.dir_name}/{video_name.stem}"
                save_path.mkdir(exist_ok=True, parents=True)

            frames, metas_frames = self.get_face(video_path, save_path, video_type)
            all_frames += frames
            metas.extend(metas_frames)

        print(f"# all frames: {all_frames}")
        if metas:
            with open(self.meta_path, "w") as f:
                json.dump(metas, f, indent=4)


if __name__ == "__main__":
    pass
