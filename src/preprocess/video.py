import cv2

from typing import Tuple, Optional, List


def create_capture(source, width: Optional[int] = None, height: Optional[int] = None):
    source = str(source).strip()
    cap = cv2.VideoCapture(source)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if cap is None or not cap.isOpened():
        print("Warning: unable to open video source: ", source)
        return None
    return cap
