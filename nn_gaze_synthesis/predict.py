import time

import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from nn_gaze_synthesis.model import EyePredModel1
from nn_gaze_synthesis.utils.datasets.dummy_dataset import DummyData
from nn_gaze_synthesis.utils.datasets.ego4d import Ego4DDataset
from nn_gaze_synthesis.utils.transforms import DEFAULT_TRANSFORMS
from nn_gaze_synthesis.utils import viz

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model):
    model.eval()

    data_set = Ego4DDataset("/srv/ssd_nvm/dataset/ego4d/ego4d/v1", sequence_length=200)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=True)

    runs = 500
    debug_show = True

    with torch.no_grad():
        for run in range(1, runs + 1):
            model.reset_history()
            for i, batch in enumerate(tqdm(data_loader)):
                data, targets = batch
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)
                
                # Move the sequence dimension to the from
                data = data.transpose(0, 1)
                targets = targets.transpose(0, 1)

                for frame, target in zip(data, targets):
                    output = model(frame)
                    canvas = viz.draw_pred_and_target(frame[0], output[0,-1], target[0])
                    if debug_show:
                        cv2.imshow(f"debug", canvas)
                        cv2.waitKey(1)
                        time.sleep(0.1)
                    else:
                        cv2.imwrite(f"viz/debug_prediction_{run:08d}_{i:03d}.jpg", canvas)

if __name__ == "__main__":
    print("Load model")
    img_size = 224
    max_len = 50
    model = EyePredModel1(img_size=img_size, token_size=128, max_len=max_len)
    model.load_state_dict(torch.load("checkpoints/model_epoch_1_step_3000.pth"))
    model.to(DEVICE)
    # Show summary
    print("Model during inference:")
    print(model)
    predict(model)
