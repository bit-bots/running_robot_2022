import time
import cv2
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from profilehooks import profile

from nn_gaze_synthesis.model import EyePredModel1
from nn_gaze_synthesis.utils.datasets.dummy_dataset import DummyData
from nn_gaze_synthesis.utils.datasets.ego4d import Ego4DDataset
from nn_gaze_synthesis.utils.transforms import DEFAULT_TRANSFORMS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@profile
def train(model):
    model.train()

    #data_set = DummyData(transform=DEFAULT_TRANSFORMS)
    data_set = Ego4DDataset("/srv/ssd_nvm/dataset/ego4d/ego4d/v1")
    data_loader = DataLoader(data_set, batch_size=1, num_workers=4, prefetch_factor=2, shuffle=True)

    lr = 0.0001
    epochs = 500
    debug_show_interval = 30
    gradient_accumulations = 12
    debug_show = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        for i, batch in enumerate(tqdm(data_loader)):
            data, targets = batch
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            if i % debug_show_interval == 0:
                for si in range(0, int(data.size(1))):
                    pred_point = tuple((output[0,si].detach().cpu().numpy() * 224).astype(int))
                    target_point = tuple((targets[0,si].detach().cpu().numpy() * 224).astype(int)) 
                    canvas = np.ascontiguousarray((data[0,si].detach().cpu()*255).permute(1,2,0).numpy().astype(np.uint8), dtype=np.uint8)
                    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                    canvas = cv2.circle(canvas, target_point, 4, (0, 0, 255), -1)
                    canvas = cv2.circle(canvas, pred_point, 4, (0, 255, 0), -1)
                    if debug_show:
                        cv2.imshow(f"debug", canvas)
                        cv2.waitKey(1)
                        time.sleep(0.1)
                    else:
                        cv2.imwrite(f"viz/debug_{i:08d}_{si:03d}.jpg", canvas)
            
            if i % gradient_accumulations == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"Loss: {loss.item()}")
        #scheduler.step()

if __name__ == "__main__":
    print("Load model")
    model = EyePredModel1(img_size=244, token_size=128, max_len=50)
    model.to(DEVICE)
    train(model)

