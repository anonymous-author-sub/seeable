from ctypes import Union
import os
import cv2
from facenet_pytorch import MTCNN
from models.retina import Retina

from PIL import Image
from tqdm import tqdm
import numpy as np
import subprocess
from typing import List

import torch
import torch.backends.cudnn as cudnn
import json

import dlib


from .utils import get_boundingbox, get_frame_types, select_samples_index


def align_face(
    predictor=None,
    box=None,
    desiredLeftEye=(0.37, 0.37),
    desiredFaceWidth=128,
    desiredFaceHeight=128,
    image=None,
):  # (0.37, 0.37)
    landmarks = torch.zeros(1, 68, 2)
    i = 0
    drect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    # Transform the prediction to numpy array
    shape = predictor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), drect)
    for j in range(0, 68):
        landmarks[i, j, :] = torch.Tensor([shape.part(j).x, shape.part(j).y])

    # code from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    left_eye_idx = [36, 37, 38, 39, 40, 41]
    right_eye_idx = [42, 43, 44, 45, 46, 47]
    # for i in range(landmarks.size()[0]): # when there is more than 1 face
    i = 0
    left_eye_pts = landmarks[i][left_eye_idx][:]
    right_eye_pts = landmarks[i][right_eye_idx][:]
    left_eye_center = left_eye_pts.mean(axis=0)
    right_eye_center = right_eye_pts.mean(axis=0)

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = torch.rad2deg(torch.atan2(dY, dX))  # - 180

    desired_right_eye_x = 1.0 - desiredLeftEye[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = desired_right_eye_x - desiredLeftEye[0]
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist
    eyesCenter = (
        int(((left_eye_center[0] + right_eye_center[0]) // 2).numpy()),
        int(((left_eye_center[1] + right_eye_center[1]) // 2).numpy()),
    )
    M = cv2.getRotationMatrix2D(eyesCenter, float(angle.numpy()), float(scale.numpy()))
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += tX - eyesCenter[0]
    M[1, 2] += tY - eyesCenter[1]
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return landmarks, output_face


def get_face(
    videoPath,
    OUTPUT_PATH,
    save_dir,
    save_face,
    method="mtcnn",
    select_nums=10,
    align=True,
):
    save_root = os.path.join(OUTPUT_PATH, save_dir)  # save faces
    if save_face:
        save_root_face = os.path.join(
            OUTPUT_PATH, save_dir + "-faces", save_face
        )  # save faces
        save_align_root_face = os.path.join(
            OUTPUT_PATH, save_dir + "-faces_align", save_face
        )  # save align faces
    v_cap = cv2.VideoCapture(videoPath)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = select_samples_index(videoPath, select_nums, v_len)

    # make dirs
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if save_face:
        if not os.path.exists(save_root_face):
            os.makedirs(save_root_face)
        if align:  # align
            if not os.path.exists(save_align_root_face):
                os.makedirs(save_align_root_face)
    if align:  # align
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    for numFrame in samples:
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, numFrame)
        _, vframe = v_cap.read()
        height, width = vframe.shape[:2]
        s = str(numFrame).zfill(3)
        try:
            if method == "retina":
                boxes, probs, points = retina.detect_faces(vframe)
            else:
                image = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                boxes, probs, points = mtcnn.detect(image, landmarks=True)
            maxi = np.argmax(boxes[:, 2] - boxes[:, 0])  # get largest face
            boxes = boxes[maxi]
            points = points[maxi]
            if align:  # align
                img_size = 256
                landmarks, aligned_face = align_face(
                    predictor=predictor,
                    box=boxes,
                    desiredFaceWidth=img_size,
                    desiredFaceHeight=img_size,
                    image=vframe,
                )  # vrne poravnano sliko enega obraza
            if save_face:
                if save_root:
                    x, y, size = get_boundingbox(boxes.flatten(), width, height)
                    cropped_face = vframe[y : y + size, x : x + size]
                    cv2.imwrite(
                        os.path.join(save_root_face, "%s.png") % s, cropped_face
                    )
                    if align:  # align
                        cv2.imwrite(
                            os.path.join(save_align_root_face, "%s.png") % s,
                            aligned_face,
                        )

            basename = os.path.splitext(os.path.basename(videoPath))[0]
            if save_root:
                outname = os.path.join(
                    save_root, video_type + "-" + basename + "_" + s + ".png"
                )
                cv2.imwrite(outname, vframe)
            # input for json file
            d[video_type + "-" + basename + "_" + s] = {
                "box": boxes.tolist(),
                "landms": points.ravel().tolist(),
                "landms_68": landmarks.squeeze().reshape(-1).tolist(),
            }
        except:
            # face not detected
            print(videoPath)
            if save_root:
                basename = os.path.splitext(os.path.basename(videoPath))[0]
                subdir = videoPath.split("/")[-2]
                outdir = os.path.join(OUTPUT_PATH, save_dir + "-nofaces/")
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                outname = outdir + subdir + "_" + basename + "_" + s + "_noface.png"
                cv2.imwrite(outname, vframe)
    v_cap.release()
    return v_len


if __name__ == "__main__":
    # Modify the following directories to yourselves

    # Celeb-DF-v2 dataset
    mode = "test"  # 'test, train'
    BASE_DIR = "/hdd2/vol1/deepfakeDatabases/original_videos/Celeb-DF-v2/"
    VIDEO_ROOT = BASE_DIR  # The base dir of DeepFake dataset
    OUTPUT_PATH = (
        "/hdd2/vol1/deepfakeDatabases/cropped_videos/Celeb-DF-v2/" + mode + "/"
    )  # Where to save cropped training faces
    TXT_PATH = (
        BASE_DIR + "Celeb-DF-v2-" + mode + "-list.txt"
    )  # the given train-list.txt or test-list.txt file
    save_face = False  # true: save faces; false: only frames with json metadata
    select_frames = "I"
    META_PATH = (
        OUTPUT_PATH + str(select_frames) + "-frames_meta.json"
    )  # where to save frames meta data  - coords of faces

    # UADFV-v2 dataset
    # BASE_DIR = '/hdd2/vol1/deepfakeDatabases/original_videos/UADFV/'        # The base dir of DeepFake dataset
    # VIDEO_ROOT = BASE_DIR        # The base dir of DeepFake dataset
    # OUTPUT_PATH = '/hdd2/vol1/deepfakeDatabases/cropped_videos/UADFV/'    # Where to save cropped faces
    # TXT_PATH = BASE_DIR+"List_of_videos.txt"    # the given train-list.txt or test-list.txt file
    # save_face=True # true: save faces; false: only frames with json metadata
    # select_frames=30
    # META_PATH = OUTPUT_PATH+str(select_frames)+"-frames_meta.json" # where to save frames meta data  - coords of faces

    with open(TXT_PATH, "r") as f:
        data = f.readlines()
    d = {}
    all_frames = 0

    # MTCNN Face detector
    mtcnn = MTCNN(device="cuda:0").eval()
    # Retina Face detector
    retina = Retina(threshold=0.9, device="cuda:0").eval()
    # cudnn.benchmark = True
    for line in tqdm(data):  # data[:100] only subset
        video_name = line[2:-1]
        video_type = line[0:1]
        video_path = os.path.join(VIDEO_ROOT, video_name)
        save_dir = None
        if OUTPUT_PATH:
            save_dir = str(select_frames) + "-frames"  # save frames with json meta data
            save_dir_face = None
            if save_face:
                save_dir_face = video_name.split(".")[0]  # save faces

        # get_face(video_path, save_dir, method='retina', select_nums=10, save_face=False) #select_nums: num of frames: 0=all, 'I'=I-frames
        frames = get_face(
            video_path,
            OUTPUT_PATH,
            save_dir,
            save_face=save_dir_face,
            method="retina",
            select_nums=select_frames,
        )  # select_nums: num of frames: 0=all, 'I'=I-frames
        all_frames += frames
    print("#all frames: " + str(all_frames))
    if d:
        with open(META_PATH, "w") as f:
            json.dump(d, f, indent=4)

