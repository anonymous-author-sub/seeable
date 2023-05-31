import json
import random
import time
import traceback
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# from src.utils import Logger
from loguru import logger
from sklearn.metrics import confusion_matrix, roc_auc_score
from src.utils.logger import format_time
from src.utils.seed import seed_everything
from torch.backends import cudnn
from torchvision.utils import save_image
from tqdm import tqdm

# from retinaface.pre_trained_models import get_model
from .inference_preprocess import extract_frames, extract_frames_level


class Inference:
    def __init__(
        self, cfg, model, dataset, detector, n_frames: int = 30, save_img: bool = False
    ):
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = dataset
        logger.info("Inference initialized")

        self.face_detector = detector  # .to(self.device)
        # get_model("resnet50_2020-07-20", max_size=2048, device=self.device)
        self.face_detector.eval()

        seed_everything()

        logger.info(f"Inference initialized, n_frames: {n_frames}")
        self.n_frames = n_frames
        self.image_size = (380, 380)
        self.save_img = save_img

    def single_inference(self, sample) -> float:
        # if self.dataset.level == "video"
        if self.dataset.level is None:  # use ffmpeg and retinaface
            path = str(sample["path"])
            # print(path)
            face_list, idx_list = extract_frames(
                path, self.n_frames, self.face_detector, image_size=self.image_size
            )
        elif self.dataset.level == "frame":  # already extracted frames
            face_list, idx_list = extract_frames_level(
                sample, image_size=self.image_size
            )
        else:
            raise NotImplementedError(f"level not implemented: {self.dataset.level}")

        # for face in face_list:
        #    print(face.shape, face.dtype, face.min(), face.max())

        with torch.no_grad():
            # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow.
            # Please consider converting the list to a single numpy.ndarray with
            # numpy.array() before converting to a tensor.
            # img = torch.tensor(face_list).to(self.device).float() / 255
            img = torch.from_numpy(np.array(face_list))
            img = img.to(device=self.device, dtype=torch.float32) / 255
            pred = self.model(img).softmax(1)[:, 1]

        if self.save_img:
            p_vid = Path("./") / str(sample["index"])
            p_vid.mkdir(exist_ok=True, parents=True)
            for i in range(img.size(0)):
                fnames = []
                fnames.extend(("index", sample["index"]))
                fnames.extend(("label", sample["label"]))
                fnames.extend(("frame", idx_list[i]))
                fnames.extend(("i", i))
                fnames.extend(("_pred", pred[i].item()))

                fname = "_".join([str(f) for f in fnames]) + ".png"
                p = p_vid / fname
                save_image(img[i, :, :, :], str(p))

        # save images
        if False:
            p = Path("./") / f"{batch}/{sample['index']}"
            p.mkdir(exist_ok=True, parents=True)
            for i in range(img.size(0)):
                save_image(img[i, :, :, :], str(p / f"{i}.png"))

        # If two or more faces are detected in a frame, the classifier is applied
        # to all faces and the highest fakeness confidence is used as the predicted
        # confidence for the frame.
        pred_list = []
        idx_img = -1
        for i in range(len(pred)):
            if idx_list[i] != idx_img:
                pred_list.append([])
                idx_img = idx_list[i]
            pred_list[-1].append(pred[i].item())

        # Once the predictions for all frames are obtained,
        # we average them to get the prediction for the video
        pred_res = np.zeros(len(pred_list))
        for i in range(len(pred_res)):
            pred_res[i] = max(pred_list[i])
        pred = pred_res.mean()

        return pred

    def inference(self):
        """we use all videos of all test sets for evaluation by setting the
        confidences to 0.5 for the videos where no face is detected in all frames
        """

        y_true = self.dataset.targets.tolist()
        y_pred = []
        indexs = []
        t0 = time.time()

        bar = tqdm(self.dataset)
        for i, sample in enumerate(bar):
            label = sample["label"]
            try:
                t0 = time.time()
                pred = self.single_inference(sample)
                dt_str = time.time() - t0
                print(f"Time: {dt_str}")
            except Exception as e:
                # traceback.print_exc()
                traceback.print_exc()
                print(e)
                pred = 0.5
            y_pred.append(pred)
            indexs.append(sample["index"])
            bar.set_description(f"label: {label} | pred: {pred:.4f}")

        # nice format time
        dt_str = format_time(time.time() - t0)
        print(f"Time: {dt_str}")

        auc = roc_auc_score(y_true, y_pred)
        print(f"AUC: {auc:.4f}")

        # cm = confusion_matrix(y_true, y_pred)
        # print(cm)

        # output ton json
        out = {
            "y_true": y_true,
            "y_pred": y_pred,
            "auc": auc,
            "indexs": indexs,
            # "cm": cm,
        }
        with open("result.json", "w") as f:
            json.dump(out, f)

    def run(self):
        print("Inference ...")
        self.inference()
