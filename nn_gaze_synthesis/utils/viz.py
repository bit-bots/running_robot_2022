import cv2
import torch
import numpy as mp

from nn_gaze_synthesis.utils.utils import gaze_postion_on_image


def show_sample(sample):
    frames = sample["video"].permute(1,2,3,0).detach().cpu()
    gazes = sample["gaze"].detach().cpu()
    for gaze, frame in zip(gazes, frames):
        x, y = gaze_postion_on_image(gaze, frame.shape).long().numpy()
        frame = cv2.cvtColor(frame.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (0,0), fx=1.0, fy=1.0)
        cv2.circle(frame, (x, y), 10, (0,0,255), -1)
        cv2.imshow("SampleViz", frame)
        cv2.waitKey(0)
