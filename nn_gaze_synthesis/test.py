import time

import cv2
import torch
from tqdm import tqdm
from torchvision.transforms import GaussianBlur

from nn_gaze_synthesis.model import EyePredModel1
from nn_gaze_synthesis.utils.datasets.ego4d import Ego4DDataset
from nn_gaze_synthesis.utils.datasets.pupilcore import PupilcoreTrackerDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model):
    model.eval()

    data_set = PupilcoreTrackerDataset("/homes/17vahl/Downloads/pupilcore_trials", sequence_length=400)

    score = 0.0

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_set)):
            model.reset_history()
            data, targets = sample[0].unsqueeze(0), sample[1].unsqueeze(0)
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Move the sequence dimension to the front
            data = data.transpose(0, 1)
            targets = targets.transpose(0, 1)

            for frame, target in zip(data, targets):
                output = model(frame)

                target = target.squeeze()
                output = output[0, -1]

                # Calculate NSS (Normalized Scanpath Saliency)
                #print(output[0] * frame.shape[2], frame.shape, target, output)
                Q_b = torch.zeros((frame.shape[2], frame.shape[3]))
                Q_b[
                    max(min(round(float(target[0] * frame.shape[2])), frame.shape[2] - 1), 0),
                    max(min(round(float(target[1] * frame.shape[3])), frame.shape[3] - 1), 0)
                ] = 1
                P = torch.zeros((frame.shape[2], frame.shape[3]))
                P[
                    max(min(round(float(output[0] * frame.shape[2])), frame.shape[2] - 1), 0),
                    max(min(round(float(output[1] * frame.shape[3])), frame.shape[3] - 1), 0)
                ] = 1
                #print(Q_b.shape, P.shape)
                P = GaussianBlur(81, sigma=10.0)(P.unsqueeze(0)).squeeze()
                cv2.imshow("Img", P.unsqueeze(2).numpy()*255)
                cv2.waitKey(0)
                score_now = (1/Q_b.sum()) * (Q_b * ((P-P.mean()) / P.std())).sum()
                print(score_now, target, output, P.std(), P.sum())
                score += score_now

    print(score / len(data_set))
                

if __name__ == "__main__":
    print("Load model")
    img_size = 224
    max_len = 50
    model = EyePredModel1(img_size=img_size, token_size=128, max_len=max_len)
    model.load_state_dict(torch.load("checkpoints/model_epoch_3_step_9500.pth"))
    model.to(DEVICE)
    # Show summary
    print("Model during inference:")
    print(model)
    evaluate(model)
