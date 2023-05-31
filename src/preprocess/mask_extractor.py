import json
import cv2

import numpy as np

from pathlib import Path

from tqdm import tqdm, trange
from tqdm.contrib.concurrent import thread_map, process_map

from typing import Tuple, Optional, List

from hydra.utils import instantiate

# speed up
from multiprocessing.pool import ThreadPool
from collections import deque

# numexpr.set_num_threads(numexpr.detect_number_of_cores())


class DummyTask:
    def __init__(self, data):
        self.data = data

    def ready(self):
        return True

    def get(self):
        return self.data


def bbox2(img):
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


class MaskExtractor:
    def __init__(
        self, cfg, dataset, root: str = None, column="target_mask_path",
    ):
        self.cfg = cfg
        self.dataset = dataset  # instantiate(dataset)
        self.column = column

        # configure
        # if root is not None else Path(cfg.dataset.root)
        self.path = Path(root)
        print("[MaskExtractor] self.path", self.path)

        # create subdir and files
        self.path_mask_bb = self.path / "mask_bb"
        self.path_mask_bb.mkdir(exist_ok=True, parents=True)

        # thread pool
        self.threaded_mode = True
        self.thread_num = cv2.getNumberOfCPUs()
        self.thread_pool = ThreadPool(processes=self.thread_num)
        self.thread_tasks = deque()

    @staticmethod
    def process_frame_old(frame):
        h, w, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, tresh = cv2.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(tresh, 1, 2)
        bbox = None
        for cnt in contours:
            x0, y0, w, h = cv2.boundingRect(cnt)
            x1, y1 = x0 + w, y0 + h
            if bbox is None:
                bbox = [x0, y0, x1, y1]
            else:
                if x0 < bbox[0]:
                    bbox[0] = x0
                if y0 < bbox[1]:
                    bbox[1] = y0
                if x1 > bbox[2]:
                    bbox[2] = x1
                if y1 > bbox[3]:
                    bbox[3] = x1
        return bbox

    @staticmethod
    def process_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, tresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        x = gray > 127
        if not x.any():
            return None
        rmin, rmax, cmin, cmax = map(int, bbox2(x))
        bbox = cmin, rmin, cmax, rmax
        return bbox

    def process_mask_video(self, index: int) -> List[str]:
        sample = self.dataset[index]
        name = sample["index"]
        mask_rel_path = sample[self.column]

        if sample["label"] == 0:
            return

        video_mask_pathlib = self.dataset.root / mask_rel_path
        if not video_mask_pathlib.is_file():
            print(
                "\n", f"[X] Target-Mask Video not found:", video_mask_pathlib,
            )
            return
        video_mask_path = str(video_mask_pathlib)

        # create outfile
        out = self.path_mask_bb / f"{name}.json"  # f"{i:0{zfill}d}"
        if out.is_file():  # skip to save time
            print("\n", f">>> Skipping:", index)
            return

        # prepare video
        cap = create_capture(video_mask_path)
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print("frame_count", frame_count)

        bboxes = []
        if False:
            for frame_idx in range(frame_count):
                # extract frame
                # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                _, frame = cap.read()
                # retrieve bbox of the frame
                bb = self.process_frame(frame)
                bboxes.append(bb)
        else:
            ret = True
            while ret:
                # Consume the queue
                while len(self.thread_tasks) > 0 and self.thread_tasks[0].ready():
                    bb = self.thread_tasks.popleft().get()
                    bboxes.append(bb)
                # Populate the queue
                if len(self.thread_tasks) < self.thread_num:
                    ret, frame = cap.read()
                    if ret:
                        if self.threaded_mode:
                            task = self.thread_pool.apply_async(
                                self.process_frame, (frame,)
                            )  # .copy()
                        else:
                            bb = self.process_frame(frame)
                            task = DummyTask(bb)
                        self.thread_tasks.append(task)

        # create output json
        frame_count = len(bboxes)
        with out.open("w") as f:
            json.dump(bboxes, f)

        return frame_count

    def run(self):
        total = len(self.dataset)
        zfill = len(str(total))
        print("[MaskExtractor] total", total)
        # -- thread --
        if False:
            if False:
                r = thread_map(self.process_mask_video, range(total), max_workers=8)
            else:
                r = process_map(
                    self.process_mask_video, range(total), max_workers=8, chunksize=1
                )
            print(type(r))
        else:
            pbar = tqdm(total=total)
            for index in range(total):
                self.process_mask_video(index)
                pbar.update(1)
                pbar.refresh()
            pbar.close()

    # https://github.com/opencv/opencv/issues/21859
