import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from nn_gaze_synthesis.model import EyePredModel1
from nn_gaze_synthesis.utils.dataset import DummyData
from nn_gaze_synthesis.utils.transforms import DEFAULT_TRANSFORMS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model):
    model.train()

    data_set = DummyData(transform=DEFAULT_TRANSFORMS)
    data_loader = DataLoader(data_set, batch_size=16, shuffle=True)

    lr = 0.0005
    epochs = 500

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        for i, batch in enumerate(tqdm(data_loader)):
            data, targets = batch
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            if i % 3 == 0:
                for i in range(5):
                    pred_point = tuple((output[0,i].detach().cpu().numpy() * 224).astype(int))
                    target_point = tuple((targets[0,i].detach().cpu().numpy() * 224).astype(int))
                    canvas = cv2.circle((data[0,i].detach().cpu()*255).permute(1,2,0).numpy().astype(np.uint8), target_point, 4, (0, 0, 255), -1)
                    canvas = cv2.circle(canvas, pred_point, 4, (0, 255, 0), -1)
                    cv2.imshow(f"debug_{i}", canvas)
                cv2.waitKey(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)
            optimizer.step()
            print(f"Loss: {loss.item()}")
        #scheduler.step()

if __name__ == "__main__":
    print("Load model")
    model = EyePredModel1(img_size=244, token_size=128, max_len=50)
    model.to(DEVICE)
    train(model)

