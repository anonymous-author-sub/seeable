import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple

# from .retina import Retina
# from facenet_pytorch import MTCNN


class Detector(torch.nn.Module):
    """
    RETINA : 
        https://github.com/biubug6/Pytorch_Retinaface

    MTCNN : https://github.com/timesler/facenet-pytorch
        pip install facenet-pytorch
        # or clone this repo, removing the '-' to allow python imports:
        git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
        # https://www.kaggle.com/code/timesler/guide-to-mtcnn-in-facenet-pytorch/notebook
    
    """

    def __init__(self, name: str, model: torch.nn.Module):
        super().__init__()
        name = name.lower()
        self.model = model.eval()
        if name == "mtcnn":
            self._call = self._call_mtcnn
        elif name == "retina":
            self._call = self._call_retina
        elif name.startswith("dlib"):
            self._call = self._call_dlib
        else:
            raise ValueError(f"Unknown detector name: {name}")
        self.name = name

    def __call__(self, vframe: np.ndarray) -> Tuple[np.ndarray, dict]:
        return self._call(vframe)

    def _call_retina(self, vframe: np.ndarray) -> Tuple[np.ndarray, dict]:
        boxes, probs, points = self.model.detect_faces(vframe)
        return boxes, probs, points

    def _call_mtcnn(self, vframe: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.
        This name is used by the forward name and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this name
        followed by the extract_face() function.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.
        Keyword Arguments:
            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
                (default: {False})
        
        Returns:
            tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
                Nx4 array of bounding boxes and a length N list of detection probabilities.
                Returned boxes will be sorted in descending order by detection probability if
                self.select_largest=False, otherwise the largest face will be returned first.
                If `img` is a list of images, the items returned have an extra dimension
                (batch) as the first dimension. Optionally, a third item, the facial landmarks,
                are returned if `landmarks=True`.
        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet_pytorch import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """
        # image = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(vframe)
        # return facial landmarks in addition to bounding boxes.
        boxes, probs, points = self.model.detect(image, landmarks=True)
        return boxes, probs, points

    def _call_dlib(self, vframe: np.ndarray) -> Tuple[np.ndarray, dict]:
        return self.model(vframe)

