import os
import sys
from typing import Dict, List

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm


def crop_face(
    img,
    landmark=None,
    bbox=None,
    margin=False,
    crop_by_bbox=True,
    abs_coord=False,
    only_img=False,
    phase="train",
):
    assert phase in ["train", "val", "test"]

    # crop face------------------------------------------
    H, W = len(img), len(img[0])
    assert landmark is not None or bbox is not None

    H, W = len(img), len(img[0])

    if crop_by_bbox:
        assert bbox is not None
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 4  # 0#np.random.rand()*(w/8)
        w1_margin = w / 4
        h0_margin = h / 4  # 0#np.random.rand()*(h/5)
        h1_margin = h / 4
    else:
        assert landmark is not None
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 8  # 0#np.random.rand()*(w/8)
        w1_margin = w / 8
        h0_margin = h / 2  # 0#np.random.rand()*(h/5)
        h1_margin = h / 5

    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == "train":
        w0_margin *= np.random.rand() * 0.6 + 0.2  # np.random.rand()
        w1_margin *= np.random.rand() * 0.6 + 0.2  # np.random.rand()
        h0_margin *= np.random.rand() * 0.6 + 0.2  # np.random.rand()
        h1_margin *= np.random.rand() * 0.6 + 0.2  # np.random.rand()
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p - x0_new, q - y0_new]
    else:
        landmark_cropped = None
    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        for i, (p, q) in enumerate(bbox):
            bbox_cropped[i] = [p - x0_new, q - y0_new]
    else:
        bbox_cropped = None

    if only_img:
        return img_cropped
    if abs_coord:
        return (
            img_cropped,
            landmark_cropped,
            bbox_cropped,
            (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1),
            y0_new,
            y1_new,
            x0_new,
            x1_new,
        )
    else:
        return (
            img_cropped,
            landmark_cropped,
            bbox_cropped,
            (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1),
        )


def extract_frames(filename, num_frames, model, image_size=(380, 380)):
    cap_org = cv2.VideoCapture(filename)

    if not cap_org.isOpened():
        print(f"Cannot open: {filename}")
        # sys.exit()
        return []

    croppedfaces = []
    idx_list = []
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(
        0, frame_count_org - 1, num_frames, endpoint=True, dtype=int
    )
    for frame_idx in frame_idxs:
        cap_org.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_org, frame_org = cap_org.read()
        if not ret_org:
            tqdm.write(f"Frame read {frame_idx} Error! : {os.path.basename(filename)}")
            break

        # ret_org, frame_org = cap_org.read()
        # height, width = frame_org.shape[:-1]
        # if not ret_org:
        #     tqdm.write(
        #         "Frame read {} Error! : {}".format(
        #             cnt_frame, os.path.basename(filename)
        #         )
        #     )
        #     break
        # if cnt_frame not in frame_idxs:
        #     continue

        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
        """
        [{'bbox': [992.42, 100.33, 1117.9, 278.48],
          'landmarks': [[1017.17, 174.22],
                        [1072.21, 163.01],
                        [1041.83, 192.92],
                        [1029.01, 231.64],
                        [1079.9, 221.22]],
          'score': 1.0},
        {'bbox': [944.89, 270.74, 959.23, 289.9],
         'landmarks': [[948.3, 277.78],
                       [954.54, 277.8],
                       [950.95, 281.09],
                       [948.85, 284.63],
                       [953.77, 284.73]],
         'score': 0.99}]
        """
        faces = model.predict_jsons(frame)
        try:
            if len(faces) == 0:
                tqdm.write(
                    f"No faces in {os.path.basename(filename)} frame {frame_idx}"
                )
                continue

            size_list = []
            croppedfaces_temp = []
            idx_list_temp = []

            bboxes = [face["bbox"] for face in faces]
            sizes = [(x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in bboxes]
            size_max = max(sizes)
            bboxes_filtered = [
                bbox
                for (bbox, size) in zip(bboxes, sizes)
                if (size >= (size_max * 0.5))
            ]
            for (x0, y0, x1, y1) in bboxes_filtered:
                bbox = np.array([[x0, y0], [x1, y1]])
                croppedfaces.append(
                    cv2.resize(
                        crop_face(
                            frame,
                            None,
                            bbox,
                            False,
                            crop_by_bbox=True,
                            only_img=True,
                            phase="test",
                        ),
                        dsize=image_size,
                    ).transpose((2, 0, 1))
                )
                idx_list.append(frame_idx)

        except Exception as e:
            print(f"error in {filename} frame {frame_idx}")
            print(e)
            continue
    cap_org.release()

    return croppedfaces, idx_list


def extract_frames_level(sample: Dict, image_size=(380, 380)):
    croppedfaces = []
    idx_list = []

    for frame in sample["frames"]:
        frame_index = frame["index"]
        frame_path = frame["path"]
        faces = frame.get("retina", [])

        # load image
        # img = frame["image"]
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect faces
        # faces = model.predict_jsons(frame)
        bboxes = [face["bbox"] for face in faces]
        if len(bboxes) == 0:
            tqdm.write(f"No faces in {frame_path} frame {frame_index}")
            continue

        try:
            # filter out faces that are too small
            sizes = [abs(x1 - x0) * abs(y1 - y0) for (x0, y0, x1, y1) in bboxes]
            size_max = float(max(sizes))
            size_limit = size_max / 2
            bbox_filtered = [
                bbox for (bbox, size) in zip(bboxes, sizes) if (size >= size_limit)
            ]

            # crop all faces in the frame
            for (x0, y0, x1, y1) in bbox_filtered:
                bbox = np.array([[x0, y0], [x1, y1]])
                img_face = crop_face(
                    img,
                    None,
                    bbox,
                    False,
                    crop_by_bbox=True,
                    only_img=True,
                    phase="test",
                )
                img_face = cv2.resize(img_face, dsize=image_size).transpose((2, 0, 1))
                croppedfaces.append(img_face)
                idx_list.append(frame_index)

        except Exception as e:
            print(f"error in {frame_path} frame {frame_index}")
            print(e)
            continue

    return croppedfaces, idx_list


def extract_face(frame, model, image_size=(380, 380)):

    faces = model.predict_jsons(frame)

    if len(faces) == 0:
        print("No face is detected")
        return []

    croppedfaces = []
    for face_idx in range(len(faces)):
        x0, y0, x1, y1 = faces[face_idx]["bbox"]
        bbox = np.array([[x0, y0], [x1, y1]])
        croppedfaces.append(
            cv2.resize(
                crop_face(
                    frame,
                    None,
                    bbox,
                    False,
                    crop_by_bbox=True,
                    only_img=True,
                    phase="test",
                ),
                dsize=image_size,
            ).transpose((2, 0, 1))
        )

    return croppedfaces
