import dlib
import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List


class DlibDetector(torch.nn.Module):
    """
    http://dlib.net/python/index.html
    """

    def __init__(
        self,
        model_path: str,
        detector_name: str = "HOG+LinearSVM",
        detector_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
        # face detector

        if detector_name == "HOG+LinearSVM":
            self.detector = dlib.get_frontal_face_detector()
            self.use_gpu = False
        elif detector_name == "MMOD_CNN":
            # http://dlib.net/files/mmod_human_face_detector.dat.bz2
            self.detector = dlib.cnn_face_detection_model_v1(detector_path)
            self.use_gpu = True
        else:
            raise ValueError(f"Unknown detector name: {detector_name}")

        # landmarks detector
        self.predictor = dlib.shape_predictor(model_path)

        self.device = device
        # self.to(device)
        if self.use_gpu:
            print("dlib.DLIB_USE_CUDA", dlib.DLIB_USE_CUDA)
            dlib.DLIB_USE_CUDA = True

    @staticmethod
    def shape_to_array(shape) -> np.ndarray:
        coords = np.zeros((shape.num_parts, 2), dtype=np.int32)
        for i, part in enumerate(shape.parts()):
            coords[i] = (part.x, part.y)
        # for i in range(shape.num_parts):
        #    coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    @staticmethod
    def shape_to_list(shape) -> List[Tuple[int, int]]:
        return [(part.x, part.y) for part in shape.parts()]

    # @staticmethod
    def rect_to_bb(self, detected) -> Tuple[int, int, int, int]:
        if self.use_gpu:
            detected = detected.rect
        x0, x1 = detected.left(), detected.right()
        y0, y1 = detected.top(), detected.bottom()
        return (x0, y0, x1, y1)

    def forward(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # face detector
        rects = self.detector(img)  # , 1)

        # landmarks detector
        boxes, probs, points = [], [], []
        for i, rect in enumerate(rects):
            shape = self.predictor(img, rect)
            landmarks = self.shape_to_list(shape)
            box = self.rect_to_bb(rect)

            boxes.append(box)
            probs.append(1.0)
            points.append(landmarks)

        boxes = np.array(boxes)
        probs = np.array(probs)
        points = np.array(points)

        return boxes, probs, points
