from typing import Optional

import cv2
import numpy as np
import torch
from torch import nn

from .data import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


class Retina(nn.Module):
    """Retina face detection module.

    [pretrained]

    mobilenetV1X0.25_pretrain.tar
    https://drive.google.com/file/d/1q36RaTZnpHVl4vRuNypoEMVWiiwCqhuD/view?usp=sharing

    resnet50-0676ba61.pth
    https://download.pytorch.org/models/resnet50-0676ba61.pth

    [Retinaface_model_v2]

    mobilenet0.25_Final.pth
    https://drive.google.com/file/d/15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1/view?usp=sharing

    Resnet50_Final.pth
    https://drive.google.com/file/d/14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW/view?usp=sharing

    Keyword Arguments:
        threshold -- confidence_threshold
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(
        self,
        cfg_name: str = "Resnet50",
        checkpoint: Optional[str] = "./weights/Resnet50_Final.pth",
        threshold: float = 0.9,
        device: Optional[str] = None,
        load_to_cpu: bool = False,
    ):
        super().__init__()
        self.cfg_name = cfg_name
        if self.cfg_name == "Resnet50":
            self.cfg = cfg_re50
        elif self.cfg_name == "mobilenet0.25":
            self.cfg = cfg_mnet
        else:
            raise Exception(f"Unsupported model : {self.cfg_name}")
        self.threshold = threshold
        self.net = RetinaFace(cfg=self.cfg, phase="test")
        if checkpoint:
            print(f"[Retina] loading checkpoint ... : {checkpoint}")
            net = self.load_model(self.net, checkpoint, load_to_cpu)
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print("Missing keys:{}".format(len(missing_keys)))
        print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
        print("Used keys:{}".format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
        return True

    def remove_prefix(self, state_dict, prefix):
        """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
        print("remove prefix '{}'".format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model, pretrained_path, load_to_cpu):
        print("Loading pretrained model from {}".format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage
            )
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
            )
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(
                pretrained_dict["state_dict"], "module."
            )
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, "module.")
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def detect_faces(self, img, origin_size=False, nms_threshold=0.4):
        img = np.float32(img)

        # testing scale
        target_size = 1600
        max_size = 1000  # 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(
                img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR
            )
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)
        probs = dets[:, 4]
        points = landms.reshape(-1, 5, 2)
        boxes = dets[:, :4]
        return boxes, probs, points

    def forward(self, img):
        """Run Retina face detection on a CV image.

        Arguments:
            img -- A CV image.


        Example:
        >>> import Retina
        >>> retina = Retina()
        >>> boxes = retina(img)
        """

        # Detect faces
        boxes = self.detect_faces(img)
        return boxes
