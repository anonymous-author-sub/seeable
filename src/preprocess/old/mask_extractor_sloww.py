import json
import cv2
from tqdm.contrib.concurrent import thread_map, process_map

from pathlib import Path
from tqdm import tqdm, trange
from typing import Tuple, Optional, List

from hydra.utils import instantiate

from .utils import get_frame_types, select_samples_index
from src.model.detector import Detector

# numexpr.set_num_threads(numexpr.detect_number_of_cores())


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

    @staticmethod
    def process_frame(frame):
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
        cap = cv2.VideoCapture(video_mask_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print("frame_count", frame_count)

        bboxes = []
        for frame_idx in range(frame_count):
            # extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()
            # retrieve bbox of the frame
            bb = self.process_frame(frame)
            bboxes.append(bb)

        # create output json
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
            for index in trange(total):
                self.process_mask_video(index)

