import subprocess
import numpy as np

from typing import Tuple, List, Union


def get_boundingbox(
    face, width: int, height: int, scale: float = 1.3, minsize: bool = None
) -> Tuple[int, int, int]:
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1, y1, x2, y2 = face[:4]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def get_frame_count(path: str) -> int:
    """
    Get the frame types of a video file : frame type ('I'/'B'/'P')
    I : Key-Frame or an Intra-frame
    B : Bi-Directional Frame
    P : Predicted Frame  
    :return: list of tuple (index, frame_type)
    """
    command = "ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format csv"
    out = subprocess.check_output(command.split() + [path]).decode()
    frame_count = out.replace("pict_type=", "").split()
    return frame_count


def get_frame_types(path: str) -> Tuple[str]:
    """
    Get the frame types of a video file : frame type ('I'/'B'/'P')
    I : Key-Frame or an Intra-frame
    B : Bi-Directional Frame
    P : Predicted Frame  
    :return: list of tuple (index, frame_type)
    """
    command = "ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1".split()
    out = subprocess.check_output(command + [path]).decode()
    frame_types = out.replace("pict_type=", "").split()
    return frame_types


def select_samples_index(
    frame_types: Tuple[str], picture_type: Union[str, int]
) -> List[int]:
    """ 
    picture_type indexs based on picture_type.
        'I'/'B'/'P' all the frames from the same type as picture_type
        int : picture_type uniformly repartitioned frames
        'all' : picture_type all the frames
    """
    frame_count = len(frame_types)
    if picture_type in ("I", "B", "P"):
        samples = [
            idx
            for idx, frame_type in enumerate(frame_types)
            if frame_type == picture_type
        ]
    elif picture_type == "all":
        samples = np.arange(frame_count)
    elif picture_type == "random":
        samples = np.random.choice(frame_count, size=frame_count, replace=False)
    elif picture_type == "first":
        samples = [0]
    elif isinstance(picture_type, int):
        num = (
            picture_type
            if (frame_count > picture_type and picture_type != 0)
            else frame_count
        )
        samples = (
            np.linspace(0, frame_count - 1, num=num, endpoint=True).round().astype(int)
        )
    else:
        raise ValueError(f"Unknown picture_type type {picture_type}")
    # if return_types:
    #    return samples, [frame_types[idx] for idx in samples]
    return samples
