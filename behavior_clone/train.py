import time

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from profilehooks import profile

from behavior_clone.model import EyePredModel1
from behavior_clone.utils.datasets.dataset import RrcDataset
from behavior_clone.utils.transforms import DEFAULT_TRANSFORMS
from behavior_clone.utils import viz

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@profile
def train(model):
    model.train()

    lr = 0.0001
    epochs = 20
    debug_show_interval = 30
    gradient_accumulations = 2
    debug_show = False

    # Get Dataset
    train_set = RrcDataset("/srv/ssd_nvm/17vahl/rrc_2022/gen_data", 10)

    # Create Dataloader
    train_data_loader = DataLoader(train_set, batch_size=64, num_workers=24, prefetch_factor=2, shuffle=True)

    # Create Optimizer, Scheduler, ...
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)
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
            loss = criterion(output.reshape(-1, 5), targets.long().reshape(-1))
            print(loss.detach().cpu())
            print(torch.argmax(output[0], axis=1).detach().cpu().tolist())
            print(targets[0].detach().cpu().long().tolist())
            # Calc gradients
            loss.backward()

            # Make a viz pintout to a window or folder showing the image, target and prediction for a sequence
            if i % debug_show_interval == 0 and False:
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

        # Step the lerning rate sheduler
        scheduler.step()
        checkpoint_name = f"checkpoints/model_epoch_{epoch}_step_{i}.pth"
        print(f"Save checkpoint '{checkpoint_name}'")
        torch.save(model.state_dict(), checkpoint_name)


if __name__ == "__main__":
    print("Load model")
    img_size = (129, 160)
    max_len = 10
    token_size=128
    model = EyePredModel1(img_size=img_size, token_size=token_size, max_len=max_len)
    model.to(DEVICE)
    # Show summary
    train(model)

