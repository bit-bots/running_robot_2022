import time

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from profilehooks import profile

from behavior_clone.model import EyePredModel1, MLPSeq
from behavior_clone.utils.datasets.dataset import RrcDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@profile
def train(model):
    model.train()

    lr = 0.0001
    epochs = 200
    debug_show_interval = 30
    gradient_accumulations = 2
    debug_show = False

    # Get Dataset
    train_set = RrcDataset("/home/florian/Projekt/bitbots/running_robot_2022/gen_data", 5)

    # Create Dataloader
    train_data_loader = DataLoader(train_set, batch_size=64, num_workers=14, prefetch_factor=2, shuffle=True)

    # Create Optimizer, Scheduler, ...
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    # Iterate over train dataset
    for epoch in range(1, epochs + 1):
        for i, batch in enumerate(tqdm(train_data_loader)):
            # Prepare data (Move to GPU, ...)
            data, targets = batch
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            # Run model
            output = model(data)
            # Calculate loss
            loss = criterion(output, targets[:, -1].long())
            print(loss.detach().cpu())
            print(torch.argmax(output, axis=1).detach().cpu().tolist())
            print(targets[:, -1].detach().cpu().long().tolist())
            # Calc gradients
            loss.backward()

            # After #gradient_accumulation steps optimize weights
            if i % gradient_accumulations == 0:
                optimizer.step()
                optimizer.zero_grad()
                # print(f"Loss: {loss.item()}")

        # Step the lerning rate sheduler
        scheduler.step()
        checkpoint_name = f"checkpoints/model_epoch_{epoch}_step_{i}.pth"
        print(f"Save checkpoint '{checkpoint_name}'")
        torch.save(model.state_dict(), checkpoint_name)


if __name__ == "__main__":
    print("Load model")
    img_size = (129, 160)
    max_len = 5
    token_size=128
    model = MLPSeq(img_size=img_size, token_size=token_size, max_len=max_len)
    model.to(DEVICE)
    # Show summary
    train(model)

