import os
import glob
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset


ACTION_CHARACTERS = ['w', 'a', 's', 'd', 'n']


class RrcDataset(Dataset):
    """Sequence data is contained in the dataset directory the following way:
    <dataset_path>
    ├── run_<timestamp>
    │   ├── frame_000000_<action>.png
    │   ├── frame_000001_<action>.png
    │   ├── ...
    ├── run_<timestamp>
    │   ├── frame_000000_<action>.png
    │   ├── frame_000001_<action>.png
    │   ├── ...
    ├── ...

    <dataset_path> is the givien path
    <action> (contained in the filename) is the choosen action of the agent after seeing the image

    Runs with fewer images than specified sequence_length will be ignored
    """
    def __init__(self, dataset_path: str, sequence_length: int):
        self.sequence_length = sequence_length

        # Collect all filenames of images for each run
        # [[... filenames of images]]
        runs = [sorted(glob.glob(f"{run_path}/*.png")) for run_path in sorted(glob.glob(f"{dataset_path}/*"))]

        # Generate filename sequences of images of given length
        self.sequences = []
        for run in runs:
            if len(run) < self.sequence_length:
                print(f"Skipping run as it contains fewer images as sequence length {self.sequence_length}: '{os.path.dirname(run[0])}'")
                continue
            for i in range(self.sequence_length - 1, len(run)):
                self.sequences.append(run[i - (self.sequence_length - 1):i + 1])

    def __getitem__(self, index):
        img_paths = self.sequences[index % len(self.sequences)]
        imgs = np.stack([cv2.imread(img_path).transpose(2, 0, 1) for img_path in img_paths])
        actions = [self._get_action_index_from_filename(img_path) for img_path in img_paths]
        return torch.from_numpy(imgs).float() / 255, torch.Tensor(actions)

    def __len__(self):
        return len(self.sequences)

    def _get_action_index_from_filename(self, filename):
        basename = os.path.basename(filename)
        action_char = basename.split('.png')[0][-1]
        try:
            action_idx = ACTION_CHARACTERS.index(action_char)
        except ValueError as e:
            print(f"Action '{action_char}' of '{filename}' is unknown. If it is correct, please consider adding it to the 'ACTION_CHARACTERS' list.")
            raise 
        return action_idx
