import os
import os.path
import json
import cv2
import numpy as np
import PIL.Image as Image


import torch
import torch.nn as nn
import torchvision.transforms as Transforms

from torch.utils.data import dataset, dataloader
from tqdm import tqdm

from src.encoder import Xception


class Model:
    def __init__(self, model_name=None):
        # init and load your model here
        model = Xception()
        model.fc = nn.Linear(2048, 1)
        thisDir = os.path.dirname(
            os.path.abspath(__file__)
        )  # use this line to find this file's dir
        if model_name == None:
            model_name = os.path.join(thisDir, "1_999_xception.ckpt")
        model.load_state_dict(torch.load(model_name))
        model.eval()
        model.cuda(0)
        self.model = model

        # determine your own batchsize based on your model size. The GPU memory is 16GB
        # relatively larger batchsize leads to faster execution.
        self.batchsize = 128

    def run(self, input_dir, json_file):
        with open(json_file, "r") as load_f:
            json_info = json.load(load_f)
        dataset_eval = DatasetFolder(input_dir, json_info)
        dataloader_eval = dataloader.DataLoader(
            dataset_eval, batch_size=self.batchsize, shuffle=False, num_workers=4
        )
        # USE shuffle=False in the above dataloader to ensure correct match between imgNames and predictions
        # Do set drop_last=False (default) in the above dataloader to ensure all images processed

        # print('Detection model inferring ...')
        prediction = []
        with torch.no_grad():  # Do USE torch.no_grad()
            for imgs in tqdm(dataloader_eval):
                imgs = imgs.to("cuda:0")
                outputs = self.model(imgs)
                preds = torch.sigmoid(outputs)
                prediction.append(preds)

        prediction = torch.cat(prediction, dim=0)
        prediction = prediction.cpu().numpy()
        prediction = prediction.squeeze().tolist()
        assert isinstance(prediction, list)
        assert isinstance(dataset_eval.imgNames, list)
        assert len(prediction) == len(dataset_eval.imgNames)

        return dataset_eval.imgNames, prediction
