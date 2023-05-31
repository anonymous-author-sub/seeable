import os
import cv2
from facenet_pytorch import MTCNN
from models.retina import Retina

from PIL import Image
from tqdm import tqdm
import numpy as np
import subprocess

import torch
import torch.backends.cudnn as cudnn
import json


from .utils import get_boundingbox, get_frame_types, select_samples_index


def get_face(videoPath, save_root, method="mtcnn", select_nums=10, save_face=True):
    v_cap = cv2.VideoCapture(videoPath)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = select_samples_index(videoPath, select_nums, v_len)

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

            if save_root and not os.path.exists(save_root):
                os.makedirs(save_root)

            if save_face:
                if save_root:
                    x, y, size = get_boundingbox(boxes.flatten(), width, height)
                    cropped_face = vframe[y : y + size, x : x + size]
                    cv2.imwrite(os.path.join(save_root, "%s.png") % s, cropped_face)
            else:
                basename = os.path.splitext(os.path.basename(videoPath))[0]
                if video_type == "1":
                    basename = videoPath.split("/")[-2] + "-" + basename
                elif video_type == "0":
                    basename = videoPath.split("/")[-3] + "-" + basename
                if save_root:
                    outname = os.path.join(
                        save_root, video_type + "-" + basename + "_" + s + ".png"
                    )
                    cv2.imwrite(outname, vframe)
                # input for json file
                d[video_type + "-" + basename + "_" + s] = {
                    "box": boxes.tolist(),
                    "landms": points.ravel().tolist(),
                }
        except:
            # face not detected
            print(videoPath)
            if save_root:
                basename = os.path.splitext(os.path.basename(videoPath))[0]
                subdir = videoPath.split("/")[-2]
                if save_face:
                    outdir = os.path.join(save_root, "../../../nofaces/")
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                else:
                    outdir = os.path.join(save_root, "../nofaces/")
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                outname = outdir + subdir + "_" + basename + "_" + s + "_noface.png"
                cv2.imwrite(outname, vframe)

    v_cap.release()
    return v_len


if __name__ == "__main__":
    # Modify the following directories to yourselves
    ## Celeb-DF-v2 dataset test - faces
    # VIDEO_ROOT = '/media/borutb/disk11/DataBase/DeepFake/Celeb-DF-v2/original_videos'        # The base dir of DeepFake dataset
    # OUTPUT_PATH = '/media/borutb/disk11/DataBase/DeepFake/Celeb-DF-v2/test/'    # Where to save cropped training faces
    ##OUTPUT_PATH = None # in case you dont want to save anything
    # TXT_PATH = "/media/borutb/disk11/DataBase/DeepFake/Celeb-DF-v2/Celeb-DF-v2-test-list.txt"    # the given train-list.txt or test-list.txt file
    # META_PATH = "/media/borutb/disk11/DataBase/DeepFake/Celeb-DF-v2/test/I-frames_meta.json" # where to save frames meta data  - coords of faces
    # save_face=True # true: save faces; false: only frames with json metadata

    # # Celeb-DF-v2 dataset - train - faces
    # mode='train'
    # BASE_DIR='/media/borutb/disk11/DataBase/DeepFake/Celeb-DF-v2/'
    # VIDEO_ROOT = BASE_DIR+'original_videos'        # The base dir of DeepFake dataset
    # OUTPUT_PATH = BASE_DIR+mode    # Where to save cropped training faces
    # #OUTPUT_PATH = None # in case you dont want to save anything
    # TXT_PATH = BASE_DIR+"Celeb-DF-v2-"+mode+"-list1.txt"    # the given train-list.txt or test-list.txt file
    # #META_PATH = BASE_DIR+mode+"/I-frames_meta.json" # where to save frames meta data  - coords of faces
    # save_face=True # true: save faces; false: only frames with json metadata

    # UADFV-v2 dataset - all - faces
    # BASE_DIR = '/media/borutb/disk11/DataBase/DeepFake/UADFV/'        # The base dir of DeepFake dataset
    # VIDEO_ROOT = BASE_DIR+'original_videos'        # The base dir of DeepFake dataset
    # OUTPUT_PATH = BASE_DIR    # Where to save cropped training faces
    # OUTPUT_PATH = None # in case you dont want to save anything
    # TXT_PATH = BASE_DIR+"List_of_videos.txt"    # the given train-list.txt or test-list.txt file
    # save_face=True # true: save faces; false: only frames with json metadata

    # UADFV-v2 dataset - all - frames
    BASE_DIR = "/media/borutb/disk11/DataBase/DeepFake/UADFV/"  # The base dir of DeepFake dataset
    VIDEO_ROOT = BASE_DIR + "original_videos"  # The base dir of DeepFake dataset
    OUTPUT_PATH = BASE_DIR  # Where to save cropped training faces
    # OUTPUT_PATH = None # in case you dont want to save anything
    TXT_PATH = (
        BASE_DIR + "List_of_videos.txt"
    )  # the given train-list.txt or test-list.txt file
    save_face = False  # true: save faces; false: only frames with json metadata
    META_PATH = (
        BASE_DIR + "30-frames_meta.json"
    )  # where to save frames meta data  - coords of faces

    # DeepfakeTIMIT dataset - all - frames
    BASE_DIR = "/media/borutb/disk11/DataBase/DeepFake/DeepfakeTIMIT/"  # The base dir of DeepFake dataset
    VIDEO_ROOT = BASE_DIR + "original_videos"  # The base dir of DeepFake dataset
    OUTPUT_PATH = BASE_DIR  # Where to save cropped training faces
    # OUTPUT_PATH = None # in case you dont want to save anything
    TXT_PATH = (
        BASE_DIR + "List_of_real_videos.txt"
    )  # the given train-list.txt or test-list.txt file
    save_face = True  # true: save faces; false: only frames with json metadata
    META_PATH = (
        BASE_DIR + "10-frames-real_meta.json"
    )  # where to save frames meta data  - coords of faces

    #   VIDEO_ROOT = '/media/borutb/Seagate Expansion Drive/Database/DeepFake/DFDC/test/'
    #   OUTPUT_PATH = './dfdc_test'
    #   TXT_PATH = "list.txt"    # the given train-list.txt file

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
            if save_face:
                save_dir = os.path.join(
                    OUTPUT_PATH, "10-frames-faces", video_name.split(".")[0]
                )  # save faces
            else:
                save_dir = os.path.join(
                    OUTPUT_PATH, "10-frames"
                )  # save frames with json meta data
        # get_face(video_path, save_dir, method='retina', select_nums=10, save_face=False) #select_nums: num of frames: 0=all, 'I'=I-frames
        frames = get_face(
            video_path, save_dir, method="retina", select_nums=10, save_face=save_face
        )  # select_nums: num of frames: 0=all, 'I'=I-frames
        # frames = get_face(video_path, save_dir, select_nums='I', save_face=save_face) #select_nums: num of frames: 0=all, 'I'=I-frames
        all_frames += frames
    print("#all frames: " + str(all_frames))
    if d:
        with open(META_PATH, "w") as f:
            json.dump(d, f, indent=4)

