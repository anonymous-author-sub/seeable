import random
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from src.augmentations.face import crop_face
from src.utils.bbox import bbox_size, landmarks_to_bbox, pairwise_iou
from src.utils.seed import seed_everything
from torchvision.io import read_image
from torchvision.utils import make_grid


def check_size(x):
    if isinstance(x, (list, tuple)):
        return len(x)
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        return x.shape
    elif isinstance(x, dict):
        return {k: check_size(v) for k, v in x.items()}
    else:
        return x


class Train:
    def __init__(self, cfg, dataset, dataloader):
        self.cfg = cfg
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # logger.info("Instantiating dataloaders")
        # self.dataloaders = instantiate(self.cfg.dataloader)
        #    self.cfg.dataloader, test_ood_dataset={"transform": self.dist_aug}
        # )

        # print(f"==> Instantiating Distribution augmentation")
        # self.dist_aug = instantiate(self.cfg.distributionaug)

    def setup(self, cfg):
        print("==> Setting up directories..")
        self.root = Path(cfg.root)
        self.root.mkdir(exist_ok=True, parents=True)
        # if self.force_init:
        #    shutil.rmtree(self.root, ignore_errors=True)

        # checkpoint directory
        self.checkpoint_dir = self.root / "raw"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # tensorboard directory
        self.tb_dir = self.root / "tb"
        self.tb_dir.mkdir(exist_ok=True)

        # json directory
        self.stats_dir = self.root / "stats"
        self.stats_dir.mkdir(exist_ok=True)

    def visu(
        self,
        img: np.ndarray,
        landmarks: np.ndarray = None,
        bbox: np.ndarray = None,
        keypoints: np.ndarray = None,
        bbox_landmarks: np.ndarray = None,
        title: Optional[str] = None,
    ):
        if isinstance(img, torch.Tensor):
            out = img.detach().cpu().numpy().transpose(1, 2, 0)
        elif isinstance(img, np.ndarray):
            out = img.copy()
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        # out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis].repeat(3, axis=2)

        if landmarks is not None:
            # draw landmarks
            for i, (x, y) in enumerate(landmarks):
                x, y = int(x), int(y)
                out = cv2.circle(out, (x, y), 1, (0, 0, 255), -1)
                if 0:
                    out = cv2.putText(
                        out,
                        str(i + 1),
                        (x + 2, y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )

            if 0:
                # draw bbox landmarks
                bbox_ld = landmarks_to_bbox(landmarks)
                x0_ld, y0_ld, x1_ld, y1_ld = list(map(int, bbox_ld))
                out = cv2.rectangle(out, (x0_ld, y0_ld), (x1_ld, y1_ld), (0, 0, 255), 1)

        if bbox_landmarks is not None:
            # draw bbox
            x0, y0, x1, y1 = list(map(int, bbox_landmarks))
            out = cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 1)

        if 1:
            # draw keypoints as losange
            if keypoints is not None:
                for i, (x, y) in enumerate(keypoints):
                    x, y = int(x), int(y)
                    out = cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
                    out = cv2.line(out, (x - 2, y), (x + 2, y), (0, 255, 0), 1)
                    out = cv2.line(out, (x, y - 2), (x, y + 2), (0, 255, 0), 1)
                    # draw text with keypoint name
                    if 0:
                        out = cv2.putText(
                            out,
                            str(i + 1),
                            (x + 2, y + 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (0, 255, 0),
                            1,
                        )

        if bbox is not None:
            # draw bbox
            x0, y0, x1, y1 = list(map(int, bbox))
            out = cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 1)

        # display
        if 0:
            face_img = cv2.resize(face_img, (256, 256), interpolation=cv2.INTER_LINEAR)

        if title is None:
            title = "face"
        title = "test"
        # cv2.destroyWindow(title)
        cv2.imshow(title, out)
        ch = cv2.waitKey(0)
        if ch == ord(" "):
            pass
        if ch == 27:  # ESC
            return 1

    def run_face(self, sample, crop=True):
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

        if crop:
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
        else:
            return self.visu(
                sample["image"],
                sample["crop"]["local"]["landmarks"],
                sample["crop"]["local"]["bbox"],
                keypoints=sample["crop"]["local"]["keypoints"],
                title=f"{video['index']}-{frame['index']}",
            )

    def single_epoch(self, epoch: int) -> None:
        print("==> Running single epoch")

        if 0:
            i = 0
            for sample in self.dataset:
                # print(sample)
                pprint(check_size(sample))
                if self.run_face(sample, crop=False):
                    return 1
                i += 1
                if i > 10:
                    break
            return
        else:

            for batch in self.dataloader:
                images, targets, samples = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                bsz = images.shape[0]

                # print("images", images.shape)
                # print("targets", targets.shape)
                # print("samples", len(samples))

                if 0:
                    Grid = make_grid(images, nrow=bsz)
                    if self.visu(Grid):
                        return 1
                else:
                    for i in range(bsz):
                        img, target, sample = images[i], targets[i], samples[i]
                        face = sample["face"]
                        # logger.debug(check_size(sample))
                        if self.visu(
                            img,
                            landmarks=face.get("landmarks"),
                            bbox=face.get("bbox"),
                            keypoints=face.get("keypoints"),
                            bbox_landmarks=face.get("bbox_landmarks"),
                        ):
                            return 1

    def run(self) -> None:

        # seed_everything(42)

        epochs = 1
        for epoch in range(epochs):
            # logger.info(f"Epoch {epoch}")
            ext = self.single_epoch(epoch)
            if ext:
                break

        cv2.destroyAllWindows()
