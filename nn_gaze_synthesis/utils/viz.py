import cv2
import torch
import numpy

from nn_gaze_synthesis.utils.utils import gaze_postion_on_image


def show_sample(sample):
    frames = sample["video"].permute(1,2,3,0).detach().cpu()
    gazes = sample["gaze"].detach().cpu()

    for gaze, frame in zip(gazes, frames):
        x, y = gaze_postion_on_image(gaze, frame.shape)
        frame = cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR)
        cv2.circle(frame, (x, y), 10, (0,0,255), -1)
        cv2.imshow("SampleViz", frame)
        cv2.waitKey(0)
