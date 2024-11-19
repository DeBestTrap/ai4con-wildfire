# %%
import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

from lib.debug import plot_rgb, plot_rgb_norm, plot_highlighted_rgb

def load_rgb_bands(dir:str, instance:str) -> np.ndarray:
    '''
    Load RGB bands
    - They are in the following order: Red, Green, Blue
    - Normalized to float values ranging from 0 to 1
    '''
    # Load bands and stack
    stacked_bands = []
    for band_file in BANDS:
        band_path = os.path.join(dir, f"{instance}{band_file}")
        with rasterio.open(band_path) as src:
            stacked_bands.append(src.read(1))
    rgb = np.dstack(stacked_bands)

    # Normalize values
    rgb_norm = (rgb / np.max(rgb)).astype(float)
    return rgb_norm

def load_burned_area_mask(dir:str, instance:str) -> np.ndarray:
    '''
    Load burned area mask
    - 0: No burned area
    - 1: Burned area
    - 249-254: Stuff we don't want (Extreme values mask, Water, Snow/Ice, Cloud, Cloud shadow)
    - 255: Fill
    '''

    # Load mask
    mask_path = os.path.join(dir, f"{instance}_BC.TIF")
    with rasterio.open(mask_path) as src:
        burned_area_mask = src.read(1)
    return burned_area_mask

def check_rgb_and_mask_exist(dir:str, instance:str) -> bool:
    # Check if the directories of band images and mask exist
    rgb_exists = True
    for band_file in BANDS:
        rgb_path = os.path.join(dir, f"{instance}{band_file}")
        rgb_exists = rgb_exists and os.path.exists(rgb_path)
    mask_path = os.path.join(dir, f"{instance}_BC.TIF")
    mask_exists = os.path.exists(mask_path)
    return rgb_exists and mask_exists

def get_rgb_and_mask(dir:str, instance:str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Return the RGB image and mask if both folders exist
    '''
    if not check_rgb_and_mask_exist(dir, instance):
        return None, None

    rgb_norm = load_rgb_bands(dir, instance)
    mask = load_burned_area_mask(dir, instance)
    return rgb_norm, mask

def extract_blocks(image, block_size):
    '''
    Extract blocks from the image
    '''
    blocks = []
    if len(image.shape) == 2:
        # If its a mask
        h, w = image.shape 
        fill_value = 255
    else:
        # If its a RGB image
        h, w, _ = image.shape

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if len(image.shape) == 3 and block.shape != (block_size, block_size, 3):
                # If its a RGB image, and the block is not the correct size, resize and fill with zeros
                resized_block = np.zeros((block_size, block_size, 3), dtype=float)
                resized_block[:block.shape[0], :block.shape[1], :] = block
                block = resized_block
            elif len(image.shape) == 2 and block.shape != (block_size, block_size):
                # If its a mask, and the block is not the correct size, resize and fill with fill_value
                resized_block = np.full((block_size, block_size), fill_value, dtype=np.uint8)
                resized_block[:block.shape[0], :block.shape[1]] = block
                block = resized_block
            blocks.append(block)
    return blocks


dir = "/home/tyler/Downloads/landsat/a"
instance = "LC09_CU_011002_20241108_20241113_02"
# instance = "LC09_CU_011003_20241108_20241113_02"
# List of band files
BANDS = [
    "_SR_B4.TIF",  # Red
    "_SR_B3.TIF",  # Green
    "_SR_B2.TIF",  # Blue
]

rgb_norm, burned_area_mask = get_rgb_and_mask(dir, instance)
# plot_burned_area_mask(burned_area_mask)
plot_highlighted_rgb(rgb_norm, burned_area_mask)

# %%
# plot dist of each channel
for i in range(3):
    plt.figure(figsize=(10, 10))
    plt.hist(rgb_norm[:, :, i].flatten()[rgb_norm[:, :, i].flatten() != 0], bins=256)
    plt.show()

# %%

# Set block size
B = 512  # Replace with desired block size

# Extract blocks from the RGB image and mask
rgb_blocks = extract_blocks(rgb_norm, B)
mask_blocks = extract_blocks(burned_area_mask, B)

# save idxes of blocks that have something
something_idxes = [i for i, rgb_block in enumerate(rgb_blocks) if np.sum(rgb_block) > 0]
save_dir = "data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for idx in something_idxes:
    print(f"Saving block {idx}")
    data = {
        "rgb": rgb_blocks[idx],
        "mask": mask_blocks[idx]
    }
    np.savez(f"{save_dir}/{instance}_block_{idx}.npz", **data)

# %%
