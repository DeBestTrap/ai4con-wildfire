# %%
import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lib.data import seperate_visible_and_infrared
from lib.debug import *

def load_image_bands(dir:str, instance:str) -> np.ndarray:
    '''
    Load image bands
    - They are in the following order: Red, Green, Blue, NIR, SWIR1, SWIR2
    - Converted to radiance values between [0, 1] float
    '''
    # Load bands and stack together to form image of (H, W, C)
    stacked_bands = []
    for band_file in BANDS:
        band_path = os.path.join(dir, f"{instance}{band_file}")
        with rasterio.open(band_path) as src:
            stacked_bands.append(src.read(1))
    bands = np.dstack(stacked_bands)

    # Convert from uint16 "Digital Number" to float [0, 1]
    radiance = bands*2.75e-05 - 0.2
    radiance = np.clip(radiance, 0, 1)

    return radiance

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

def check_image_and_mask_exist(dir:str, instance:str) -> bool:
    '''
    Check if the directories of band images and mask exist
    '''
    image_exists = True
    for band_file in BANDS:
        image_path = os.path.join(dir, f"{instance}{band_file}")
        image_exists = image_exists and os.path.exists(image_path)
    mask_path = os.path.join(dir, f"{instance}_BC.TIF")
    mask_exists = os.path.exists(mask_path)
    return image_exists and mask_exists

def get_image_and_mask(dir:str, instance:str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Return the image and mask if both folders exist
    '''
    if not check_image_and_mask_exist(dir, instance):
        return None, None

    image = load_image_bands(dir, instance)
    mask = load_burned_area_mask(dir, instance)
    return image, mask

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
        # If its an image
        h, w, c = image.shape

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if len(image.shape) == 3 and block.shape != (block_size, block_size, c):
                # If its an image, and the block is not the correct size, resize and fill with zeros
                resized_block = np.zeros((block_size, block_size, c), dtype=float)
                resized_block[:block.shape[0], :block.shape[1], :] = block
                block = resized_block
            elif len(image.shape) == 2 and block.shape != (block_size, block_size):
                # If its a mask, and the block is not the correct size, resize and fill with fill_value
                resized_block = np.full((block_size, block_size), fill_value, dtype=np.uint8)
                resized_block[:block.shape[0], :block.shape[1]] = block
                block = resized_block
            blocks.append(block)
    return blocks

def extract_blocks_from_dir(dir, instance, block_size, extract_only_burned=False) -> bool:
    '''
    Extracts smaller blocks from the image and mask and saves them to a directory

    The images are saved in (H, W, C) and as a float from [0, 1]
    '''
    image, burned_area_mask = get_image_and_mask(dir, instance)

    # Return if both mask and image don't exist
    if image is None or burned_area_mask is None:
        return False

    if DEBUG:
        rgb, infared = seperate_visible_and_infrared(image)
        # plot_highlighted_rgb_and_mask(rgb, burned_area_mask, only_burned=True)
        plot_rgb(infared)
        # plot_highlighted_rgb_and_mask(infared, burned_area_mask, only_burned=True)
        plot_burned_area_mask(burned_area_mask)

        # plot dist of each channel
        fig, ax = plt.subplots(2, 3, figsize=(10, 5))
        for i, name in enumerate(BANDS):
            ax[i // 3, i % 3].set_title(name)
            ax[i // 3, i % 3].hist(image[:, :, i].flatten()[image[:, :, i].flatten() != 0], bins=256)
        fig.tight_layout()
        fig.show()

    # Extract blocks from the image and mask
    image_blocks = extract_blocks(image, block_size)
    mask_blocks = extract_blocks(burned_area_mask, block_size)

    # Make a directory to save the blocks
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # remove blocks with no data in the image
    images_w_stuff_idxes = [i for i, image_block in enumerate(image_blocks) if np.sum(image_block) > 0]

    # remove blocks with no burned area in the mask
    masks_w_burned_area_idxes = [i for i, mask_block in enumerate(mask_blocks) if np.sum(mask_block == 1) > 0]
    if extract_only_burned:
        save_idxes = [i for i in images_w_stuff_idxes if i in masks_w_burned_area_idxes]
    else:
        save_idxes = images_w_stuff_idxes

    # visualize the blocks
    # if DEBUG:
        # for idx in save_idxes:
        #     rgb, infared = seperate_visible_and_infrared(image_blocks[idx])
        #     plot_rgb(rgb)
        #     plot_rgb(infared)
        #     plot_burned_area_mask(mask_blocks[idx])

    # save the blocks
    for idx in save_idxes:
        data = {
            "image": image_blocks[idx],
            "mask": mask_blocks[idx]
        }
        np.savez(f"{save_dir}/{instance}_block_{idx}.npz", **data)
    return True


# List of band files
BANDS = [
    "_SR_B4.TIF",  # Red
    "_SR_B3.TIF",  # Green
    "_SR_B2.TIF",  # Blue
    "_SR_B7.TIF",  # SWIR2 (Shortwave Infrared 2100nm-2300nm)
    "_SR_B5.TIF",  # NIR (Near Infrared)
    "_SR_B6.TIF",  # SWIR1 (Shortwave Infrared 1560nm-1660nm)
]

# Parameters
DEBUG = False
dir = "/mnt/csdrive/landsat/combined/"
save_dir = "data_infrared/"

# Search for instances
instances = set()
for file in os.listdir(dir):
    if file.endswith(".TIF"):
        instance = "_".join(file.split("_")[0:6])
        instances.add(instance)
print(f"Found {len(instances)} instances")

# Iterate over instances and extract blocks into data folder
for instance in tqdm(instances):
    extract_blocks_from_dir(dir, instance, block_size=256, extract_only_burned=True)




'''
# ignore this, this was temporary code for copying instances that have both an image and mask
# just in case i need to do this again
import shutil
def copy_all_files(instance, source_dir, target_dir):
    for file in os.listdir(source_dir):
        if instance in file:
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)
            shutil.copy2(source_path, target_path)


ctr = 0
new_dir = "/mnt/csdrive/landsat/test/"
for instance in instances:
    if check_rgb_and_mask_exist(dir, instance):
        copy_all_files(instance, dir, new_dir)
        ctr += 1

    if ctr > 10:
        break
'''