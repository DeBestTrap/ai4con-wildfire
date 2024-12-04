import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.data import remove_filler_values, rgb_float_to_uint8


def convert_to_numpy(tensor: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to a numpy array

    Will raise an error if not a numpy array or a torch tensor
    """
    if type(tensor) == torch.Tensor:
        return tensor.numpy()
    elif type(tensor) == np.ndarray:
        return tensor
    else:
        raise TypeError("tensor must be a numpy array or a torch tensor")


'''
-------------------- Ploting functions for masks
'''
def get_highlighted_mask(mask: np.ndarray, only_burned=False) -> np.ndarray:
    """
    Key:
        0:   No burned area      | (Black)
        1:   Burned area         | (Red)
        249: Extreme values mask
        250: Extreme values mask
        251: Water               | (Blue)
        252: Snow/Ice            | (Light Blue)
        253: Cloud               | (White)
        254: Cloud shadow        | (Dark gray)
        255: Fill                | (Green)
    """
    color_map = {
        0: (0, 0, 0),  # Black
        1: (255, 0, 0),  # Red
        249: (0, 0, 0),  # Extreme values mask
        250: (0, 0, 0),  # Extreme values mask
        251: (0, 0, 255),  # Blue
        252: (0, 255, 255),  # Light Blue
        253: (255, 255, 255),  # White
        254: (128, 128, 128),  # Dark gray
        255: (0, 255, 0),  # Green
    }
    # Convert mask to an RGB image
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        if only_burned and class_id != 1:
            continue
        color_mask[mask == class_id] = color

    return color_mask


def plot_mask(mask):
    # Get the color mask
    color_mask = get_highlighted_mask(mask)

    # Display the color mask
    plt.figure(figsize=(10, 10))
    plt.imshow(color_mask)
    plt.axis("off")
    plt.show()


def plot_burned_area_mask(mask):
    mask = convert_to_numpy(mask)

    plt.figure(figsize=(10, 10))
    plt.imshow(remove_filler_values(mask), cmap="gray")
    plt.axis("off")
    plt.show()


'''
-------------------- Ploting functions for RGB images
'''
def plot_rgb_float(rgb_norm: np.ndarray | torch.Tensor) -> None:
    """
    Plot a RGB image stored with float

    Will raise an error if the image is not in the shape of (H, W, 3)
    """
    plot_rgb(rgb_float_to_uint8(rgb_norm))


def plot_rgb(rgb: np.ndarray | torch.Tensor) -> None:
    """
    Plot a RGB image stored with uint8

    Will raise an error if the image is not in the shape of (H, W, 3)
    """
    if type(rgb) == torch.Tensor:
        rgb = rgb.numpy()
    if rgb.shape[2] != 3:
        assert False, f"RGB image must be in the shape of (H, W, 3): {rgb.shape}"

    plt.figure(figsize=(20, 20))
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()

def denormalize(rgb: np.ndarray) -> np.ndarray:
    """
    Undo normalization for a tensor image and convert it to NumPy for plotting.
    Args:
        rgb: Normalized image tensor of shape (H, W, C).
    Returns:
        np.ndarray: Denormalized image as a NumPy array in the range [0, 1].
    """
    # Define mean and std used for normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Denormalize
    rgb = (rgb * std) + mean

    # Not sure why we need to multiply by 255
    rgb = rgb*255

    # Clip to [0, 1] range
    rgb = np.clip(rgb, 0, 1)
    return rgb


def plot_preprocessed_rgb(rgb: np.ndarray | torch.Tensor, normalized: bool) -> None:
    """
    If the RGB image is preprocessed, undo the transformation
    """
    rgb = convert_to_numpy(rgb)
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if normalized:
        rgb = denormalize(rgb)
    plot_rgb_float(rgb)



'''
-------------------- Ploting functions for combined RGB images and masks
'''
def plot_highlighted_rgb_and_mask(
    rgb: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    only_burned: bool = False,
) -> None:
    # Convert to numpy
    rgb = convert_to_numpy(rgb)
    mask = convert_to_numpy(mask)

    # Create an RGB composite with masked areas highlighted
    highlighted_rgb = rgb_float_to_uint8(rgb.copy())
    # Get the color mask
    color_mask = get_highlighted_mask(mask, only_burned=only_burned)

    # Highlight the masked areas in the RGB composite
    highlighted_rgb[color_mask != (0, 0, 0)] = color_mask[color_mask != (0, 0, 0)]

    # Plot
    plot_rgb(highlighted_rgb)

def plot_highlighted_rgb_and_mask_preprocessed(
    rgb: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    normalized: bool,
    only_burned: bool = False,
) -> None:
    # Convert to numpy
    rgb = convert_to_numpy(rgb)
    mask = convert_to_numpy(mask)
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if normalized:
        rgb = denormalize(rgb)
    plot_highlighted_rgb_and_mask(rgb, mask, only_burned=only_burned)