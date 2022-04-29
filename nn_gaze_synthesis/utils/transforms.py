import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class SequencifyTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        images, labels = zip(*x)
        images = np.stack(images)
        labels = np.array(labels)
        return (images, labels)


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, x):
        images, labels = x
        images = torch.from_numpy(images, ).permute(0, 3, 1, 2).float()
        labels = torch.from_numpy(labels).float()
        return (images, labels)


class Normalize:
    def __init__(self):
        pass

    def __call__(self, x):
        images, labels = x
        images = images / 255
        # TODO normalize labels
        labels = labels / 224
        return (images, labels)


DEFAULT_TRANSFORMS = transforms.Compose([
    SequencifyTransform(),
    ToTensor(),
    Normalize()
])
