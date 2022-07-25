import cv2
from detect import *
import time
import numpy as np
import math
from numpy import random
from pathlib import Path
import torch
import argparse


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGTHS = ROOT / 'weights'


def detect_video(url_video=None, path_model=None, flag_save=False, fps=None, name_video='video.avi'):

    # ******************************** LOAD MODEL *************************************************
    # load model detect yolov5
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    y7_model = Y7Detect(weights=path_model)
    class_name = y7_model.class_names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_name]

    # ********************************** READ VIDEO **********************************************
    if url_video == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(url_video)

    # get size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if frame_height > 1080 and frame_width > 1920:
        frame_width = 1920
        frame_height = 1080
    # get fps of camera
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    # save video
    if flag_save is True:
        video_writer = cv2.VideoWriter(name_video,
                                       cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

    # ******************************** REAL TIME *********************************************
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) == ord('q'):
            break
        h, w, _ = frame.shape
        if h > 1080 and w > 1920:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape

        # ************************ DETECT YOLOv5 ***************************************
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox, label, score, label_id = y7_model.predict(image)
        for idx, box in enumerate(bbox):
            icolor = class_name.index(label[idx])
            draw_boxes(frame, box, label[idx], round(score[idx]*100), colors[icolor])

        # ******************************************** SHOW *******************************************
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        fps = int(1 / (time.time() - start))
        cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('video', frame)
        cv2.waitKey(1)

        if flag_save is True:
            video_writer.write(frame)

    cap.release()
    if flag_save is True:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face On Video')
    parser.add_argument("-fn", "--file_name", help="video file name or rtsp", default='', type=str)
    parser.add_argument("-op", "--option", help="if save video then choice option = 1", default=False, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='face_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=20, help="FPS of output video", type=int)
    args = parser.parse_args()


    # MODEL YOLOV5
    path_models = WEIGTHS / 'yolov7.pt'
    # PATH VIDEO
    url = '/home/duyngu/Downloads/video_test/TownCentre.mp4'
    source = args.file_name
    cv2.namedWindow('video')
    # if run  as terminal, replace url = source
    detect_video(url_video=url, path_model=path_models,
              flag_save=args.option, fps=args.fps, name_video=args.output)
