import numpy as np
import torch
import albumentations as A
from typing import Tuple


MASK_FILLER_VALUES = [
    249,  # Extreme values mask
    250,  # Extreme values mask
    251,  # Water
    252,  # Snow/Ice
    253,  # Cloud
    254,  # Cloud shadow
    255,  # Fill
]


def remove_filler_values(mask):
    for value in MASK_FILLER_VALUES:
        mask[mask == value] = 0
    return mask


def rgb_float_to_uint8(rgb: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    # Convert float values to uint8
    if type(rgb) == np.ndarray:
        return (rgb * 255).astype(np.uint8)
    elif type(rgb) == torch.Tensor:
        return (rgb * 255).type(torch.uint8)
    else:
        raise TypeError("rgb must be a numpy array or a torch tensor")


def seperate_visible_and_infrared(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Seperate the visible and infrared bands
    """
    return image[:, :, :3], image[:, :, 3:]


class InverseNormalize(A.ImageOnlyTransform):
    def __init__(self, mean, std, always_apply=True, p=1.0):
        super(InverseNormalize, self).__init__(always_apply=always_apply, p=p)
        self.mean = np.array(mean)
        self.std = np.array(std)

    def apply(self, img, **params):
        # img should be float32 in [0,1] range after A.Normalize
        img = (img * self.std) + self.mean
        img = np.clip(img, 0, 1)
        return img
