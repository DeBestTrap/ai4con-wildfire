import numpy as np
import torch

MASK_FILLER_VALUES = [
    249, # Extreme values mask
    250, # Extreme values mask
    251, # Water
    252, # Snow/Ice
    253, # Cloud
    254, # Cloud shadow
    255, # Fill
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