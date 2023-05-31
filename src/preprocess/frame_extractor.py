# import concurrent.futures
# import json
# import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from hydra.utils import instantiate
from loguru import logger
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

from .utils import get_boundingbox, get_frame_types, select_samples_index
from .video import create_capture

# numexpr.set_num_threads(numexpr.detect_number_of_cores())


class FrameExtractor:
    def __init__(
        self,
        cfg,
        dataset,
        root: str = None,
        picture_type: str = "I",
        extension: str = ".png",
    ):
        self.dataset = dataset  # instantiate(dataset)
        # configure the frames to be extracted
        if not isinstance(picture_type, (str, int)):
            raise Exception("[FrameExtractor] picture_type must be a string or an int")
        if isinstance(picture_type, str):
            if picture_type.isnumeric():
                picture_type = int(picture_type)
                if picture_type <= 0:
                    raise Exception(
                        "[FrameExtractor] picture_type must be a positive int"
                    )
            elif picture_type in ["I", "B", "P"] + ["all", "random", "first"]:
                picture_type = picture_type
            else:
                raise Exception("[FrameExtractor] picture_type : int, I, B, P, or all")
        self.picture_type = picture_type
        self.extension = extension

        # configure
        self.path = Path(root)  # if root is not None else Path(cfg.dataset.root)
        print("[FrameExtractor] self.path", self.path)

        # create subdir and files
        self.path_images = self.path / "images"
        self.path_metas = self.path / "metas"
        self.path_metas.mkdir(exist_ok=True, parents=True)

    def process_video(self, index: int) -> List[str]:
        sample = self.dataset[index]
        name = sample["index"]
        video_path: str = str(sample["path"])
        video_pathlib = Path(video_path)
        if not video_pathlib.is_file():
            msg = f"video not found {video_pathlib}"
            logger.error(msg)
            raise Exception
            return

        # get frame types
        meta_file = self.path_metas / f"{name}.txt"
        if meta_file.is_file():
            with meta_file.open("r") as f:
                frame_types = f.read().split(",")
        else:
            frame_types = get_frame_types(video_path)
            with meta_file.open("w") as f:
                f.write(",".join(frame_types))

        # select which frames index to extract
        samples = select_samples_index(frame_types, self.picture_type)
        frame_count = len(samples)

        # prepare video
        cap = create_capture(video_path)
        if cap is None:
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if len(frame_types) != frame_count:
            msg = (
                f"Video length: {video_pathlib.name} {len(frame_types)} {frame_count}",
            )
            logger.warning(msg)
            return

        # create subdir
        output_dir_name = str(name)  # f"{i:0{zfill}d}"
        output_dir = self.path_images / output_dir_name
        output_dir.mkdir(exist_ok=True, parents=True)

        for frame_idx in samples:
            # for frame_idx, frame_type in zip(samples, frame_types):
            path = output_dir / f"{frame_idx}.{self.extension}"
            if path.is_file():  # skip to gain time
                continue

            # extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()

            # retrieve information
            # height, width = frame.shape[:2]

            # save frame
            # p.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(path), frame)

        return samples

    """
    def run(self):
        total = len(self.dataset)
        zfill = len(str(total))
        queries = self.dataset.folds
        queries = [None] if queries is None else queries

        print("[FrameExtractor] total", total)
        print("[FrameExtractor] queries", queries)
        print("[FrameExtractor] splits", self.dataset.splits)

        # for query in tqdm(queries, desc="outer", position=0):
        for query in queries:
            print("->", query)

            # retrieve the subset based on the query
            subset = self.dataset.get_fold_subset(query)
            if subset is None:
                subset = range(total)
            else:
                subset = sorted(subset)
            print("-> subset :", len(subset))

            # -- thread --
            if False:
                r = thread_map(self.process_video, subset, max_workers=8)
                print(type(r))
            else:
                for index in tqdm(subset):
                    self.process_video(index)
    """

    def run(self, mode="thread"):
        """
        https://vuamitom.github.io/2019/12/13/fast-iterate-through-video-frames.html


        """
        total = len(self.dataset)
        zfill = len(str(total))
        print("[FrameExtractor] total", total)
        # -- thread --
        if mode == "thread":  # 2
            r = thread_map(self.process_video, range(total), max_workers=8)
            print(type(r))
        elif mode == "process":  # 1
            r = process_map(
                self.process_video, range(total), max_workers=8, chunksize=1
            )
            print(type(r))
        else:  # 3
            pbar = tqdm(total=total)
            for index in range(total):
                self.process_video(index)
                pbar.update(1)
                pbar.refresh()
            pbar.close()
