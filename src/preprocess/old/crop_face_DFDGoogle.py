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
from os import listdir
import math


from .utils import get_boundingbox, get_frame_types, select_samples_index


def get_face(
    VIDEO_ROOT,
    video_name,
    OUTPUT_PATH,
    maskPath,
    save_dir,
    method="mtcnn",
    select_nums=10,
    save_face=True,
):
    videoPath = os.path.join(VIDEO_ROOT, video_name)
    save_root = os.path.join(OUTPUT_PATH, save_dir)  # save faces
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

            w = boxes[:, 2] - boxes[:, 0]
            maxi = np.argmax(w)  # get largest face (width)
            # from 80% largest face take this with max probability
            largest = w >= w[maxi] * 0.5
            probi = np.argmax(probs[largest])  # get face with max probability

            # make dirs
            if save_root and not os.path.exists(save_root):
                os.makedirs(save_root)

            if save_root:
                save_root_face = os.path.join(
                    "-".join(save_root.split("-")[:-1]) + "-faces",
                    video_name.split(".")[0],
                )
                if not os.path.exists(save_root_face):
                    os.makedirs(save_root_face)

            # save face with largest probability
            boxes1 = boxes[largest][probi]
            points1 = points[largest][probi]
            if save_root:
                x, y, size = get_boundingbox(boxes1.flatten(), width, height)
                cropped_face = vframe[y : y + size, x : x + size]
                cv2.imwrite(os.path.join(save_root_face, "%s.png") % s, cropped_face)

            # more faces and mask
            if boxes[largest].shape[0] > 1 and maskPath:

                # get mask
                # maskPath = videoPath.split('/')
                # maskPath[-3] = 'masks'
                # maskPath = os.path.join('/',*maskPath)

                # v primeru če je video maske krajši vzame najbližji frame in izračuna razdaljo med obrazi
                # vzame obraz najbližje maski
                m_cap = cv2.VideoCapture(maskPath)
                m_len = int(m_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if v_len != m_len:
                    if numFrame > m_len:
                        numFrame = 0

                m_cap.set(cv2.CAP_PROP_POS_FRAMES, numFrame)
                _, mframe = m_cap.read()
                mframe = cv2.cvtColor(mframe, cv2.COLOR_BGR2GRAY)
                # find center
                # calculate moments of binary image
                M = cv2.moments(mframe)
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # mframe = mframe/255
                min_dist = width + height

                a = np.array((cX, cY))

                # izračunamo razdalje obrazov do centra maske
                for i, iboxes in enumerate(boxes[largest]):
                    x1, y1, size1 = get_boundingbox(iboxes.flatten(), width, height)
                    b = np.array((x1 + size1 / 2, y1 + size1 / 2))
                    dist = np.linalg.norm(a - b)  # Euclidean distance
                    if dist < min_dist:
                        min_dist = dist
                        x, y, size = x1, y1, size1
                        boxes1 = iboxes
                        points1 = points[largest][i]

                cropped_face = vframe[y : y + size, x : x + size]

                #                if save_root:
                #                    save_root_face2 = os.path.join('-'.join(save_root.split('-')[:-1])+'-multifaces',video_name.split('.')[0])
                #                    if not os.path.exists(save_root_face2):
                #                        os.makedirs(save_root_face2)
                #                cv2.imwrite(os.path.join(save_root_face2,"%s.png") % s, cropped_face)
                cv2.imwrite(
                    os.path.join(save_root_face, "%s.png") % s, cropped_face
                )  # overwrite
                m_cap.release()

            if save_root:
                basename = os.path.splitext(os.path.basename(videoPath))[0]
                if save_root:
                    outname = os.path.join(
                        save_root, video_type + "-" + basename + "_" + s + ".png"
                    )
                    cv2.imwrite(outname, vframe)
                # input for json file
                d[video_type + "-" + basename + "_" + s] = {
                    "box": boxes1.tolist(),
                    "landms": points1.ravel().tolist(),
                }
        except:
            # face not detected
            print(videoPath)
            if save_root:
                basename = os.path.splitext(os.path.basename(videoPath))[0]
                subdir = videoPath.split("/")[-2]
                outdir = os.path.join(OUTPUT_PATH, "nofaces", df)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                outname = os.path.join(
                    outdir, subdir + "_" + basename + "_" + s + "_noface.png"
                )
                cv2.imwrite(outname, vframe)
    v_cap.release()
    return v_len


if __name__ == "__main__":
    # Modify the following directories to yourselves
    BASE_DIR = "/media/borutb/disk11/DataBase/DeepFake/DFDGoogle/"  # The base dir of DeepFake dataset
    VIDEO_ROOT = BASE_DIR + "original_videos"  # The base dir of DeepFake dataset
    OUTPUT_PATH = BASE_DIR  # Where to save cropped training faces

    #'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'original_sequences'
    # for df in ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'original_sequences']:
    "DeepFakeDetection", "original_sequences"
    for df in ["original_sequences"]:

        TXT_PATH = (
            BASE_DIR + df + ".txt"
        )  # the given train-list.txt or test-list.txt file
        save_face = False  # true: save faces; false: only frames with json metadata
        META_PATH = (
            BASE_DIR + "I-frames-" + df + "_meta.json"
        )  # where to save frames meta data  - coords of faces

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
                save_dir = os.path.join(
                    OUTPUT_PATH, "I-frames-" + df
                )  # save frames with json meta data
            # if masks exist
            maskpath = video_path.split("/")
            maskpath[-3] = "masks"
            maskpath = os.path.join("/", *maskpath)
            # if masks not exist (in original_sequences) take any face
            if not os.path.exists(maskpath):
                maskpath = None

            frames = get_face(
                VIDEO_ROOT,
                video_name,
                OUTPUT_PATH,
                maskpath,
                save_dir,
                method="retina",
                select_nums="I",
                save_face=save_face,
            )  # select_nums: num of frames: 0=all, 'I'=I-frames
            all_frames += frames
        print("#all frames: " + str(all_frames))
        if d:
            with open(META_PATH, "w") as f:
                json.dump(d, f, indent=4)

