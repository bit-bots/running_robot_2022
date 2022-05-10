import time

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from profilehooks import profile

from nn_gaze_synthesis.model import EyePredModel1
from nn_gaze_synthesis.utils.datasets.dummy_dataset import DummyData
from nn_gaze_synthesis.utils.datasets.ego4d import Ego4DDataset
from nn_gaze_synthesis.utils.transforms import DEFAULT_TRANSFORMS
from nn_gaze_synthesis.utils import viz

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@profile
def train(model):
    model.train()

    # Params TODO config file, ...
    lr = 0.0001
    epochs = 500
    debug_show_interval = 30
    gradient_accumulations = 12
    debug_show = False
    evaluation_interval = 500 # Batches
    validation_split = 0.02

    # Get Dataset
    #data_set = DummyData(transform=DEFAULT_TRANSFORMS)
    data_set = Ego4DDataset("/srv/ssd_nvm/dataset/ego4d/ego4d/v1")

    # Split into train / validation partitions
    n_val = int(round(len(data_set) * validation_split))
    n_train = len(data_set) - n_val
    train_set, val_set = random_split(data_set, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create Dataloader
    train_data_loader = DataLoader(train_set, batch_size=1, num_workers=4, prefetch_factor=2, shuffle=True)
    val_data_loader = DataLoader(val_set, batch_size=1, num_workers=4, prefetch_factor=2, shuffle=False)

    # Create Optimizer, Scheduler, ...
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)
    criterion = nn.MSELoss()

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
            loss = criterion(output, targets)
            # Calc gradients
            loss.backward()

            # Make a viz pintout to a window or folder showing the image, target and prediction for a sequence
            if i % debug_show_interval == 0:
                # Iterate over sequence dimension
                for si in range(0, int(data.size(1))):
                    # Draw both on the image
                    canvas = viz.draw_pred_and_target(data[0,si], output[0,si], targets[0,si])
                    # Show or save image
                    if debug_show:
                        cv2.imshow(f"debug", canvas)
                        cv2.waitKey(1)
                        time.sleep(0.1)
                    else:
                        cv2.imwrite(f"viz/debug_{i:08d}_{si:03d}.jpg", canvas)

            # After #gradient_accumulation steps optimize weights
            if i % gradient_accumulations == 0:
                optimizer.step()
                optimizer.zero_grad()
                # print(f"Loss: {loss.item()}")

            # Evaluate network and make a checkpoint
            if i % evaluation_interval == 0 and i != 0:
                print("Evaluate...")
                # TODO evaluation
                # Save model
                checkpoint_name = f"checkpoints/model_epoch_{epoch}_step_{i}.pth"
                print(f"Save checkpoint '{checkpoint_name}'")
                torch.save(model.state_dict(), checkpoint_name)

        # Step the lerning rate sheduler
        scheduler.step()

if __name__ == "__main__":
    print("Load model")
    img_size = 224
    max_len = 50
    token_size=128
    model = EyePredModel1(img_size=img_size, token_size=token_size, max_len=max_len)
    model.to(DEVICE)
    # Show summary
    print("Model during inference:")
    print(model)
    train(model)

