import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from nn_gaze_synthesis.model import EyePredModel1
from nn_gaze_synthesis.utils.dataset import DummyData
from nn_gaze_synthesis.utils.transforms import DEFAULT_TRANSFORMS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model):
    model.train()

    data_set = DummyData(transform=DEFAULT_TRANSFORMS)
    data_loader = DataLoader(data_set, batch_size=2, shuffle=True)

    lr = 0.005
    epochs = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        for i, batch in tqdm(enumerate(data_loader)):
            data, targets = batch
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)
            optimizer.step()
            print(f"Loss: {loss.item()}")
        scheduler.step()

if __name__ == "__main__":
    print("Load model")
    model = EyePredModel1(img_size=244, token_size=128, max_len=5000)
    model.to(DEVICE)
    train(model)

