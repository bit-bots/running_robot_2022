import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_closest(array: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Match two numpy arrays (closest elements).
    Returns indices.
    """
    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    return idxs

def gaze_postion_on_image(gaze, image_dim):
    if isinstance(gaze, torch.Tensor):
        return gaze * torch.tensor(image_dim[:2])
    elif isinstance(gaze, np.ndarray):
        return gaze * np.asarray(image_dim[:2])
