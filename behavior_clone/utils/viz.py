import cv2
import numpy as np


def show_sample(sample):
    """
    Displays a sample from a dataset.
    """
    frames = sample[0].permute(1,2,3,0).detach().cpu()
    gazes = sample[1].detach().cpu()
    for gaze, frame in zip(gazes, frames):
        x, y = gaze_postion_on_image(gaze, frame.shape).long().numpy()
        frame = cv2.cvtColor(frame.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (0,0), fx=1.0, fy=1.0)
        cv2.circle(frame, (x, y), 10, (0,0,255), -1)
        cv2.imshow("SampleViz", frame)
        cv2.waitKey(0)


def draw_pred_and_target(frame, pred, target):
    """
    Draws a target as well as the prediction on the corresponding frame
    :param frame: Input frame as a tensor
    :param pred: Network prediction as a fraction of the image dimension (tensor)
    :param target: Target value as a fraction of the image dimension (tensor)
    :returns: Image as a BGR CV2 NumPy array-
    """
    pred_point = tuple((pred.detach().cpu().numpy() * 224).astype(int))
    target_point = tuple((target.detach().cpu().numpy() * 224).astype(int)) 
    canvas = np.ascontiguousarray(
        (frame.detach().cpu() * 255).permute(1,2,0).numpy().astype(np.uint8), 
        dtype=np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    canvas = cv2.circle(canvas, target_point, 4, (0, 0, 255), -1)
    canvas = cv2.circle(canvas, pred_point, 4, (0, 255, 0), -1)
    return canvas
