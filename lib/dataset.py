import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from lib.data import remove_filler_values, rgb_float_to_uint8

DEFAULT = A.Compose([
    ToTensorV2() 
])

class SatteliteImageDataset(Dataset):
    def __init__(self, data_dir: str, preprocess_transform: A.Compose=DEFAULT, transform: A.Compose=None) -> None:
        self.data_dir = data_dir
        self.preprocess_transform = preprocess_transform
        self.augment_transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('npz')]

    def __len__(self):
        return len(self.file_list)
    
    def apply_transforms(self, transform, image, mask):
        '''
        Apply the transform to the image and mask
        (albumentations transforms work with both images and masks)

        Returns the transformed image and mask
        '''
        transformed = transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        return image, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load the data
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        rgb = data['rgb']
        mask = data['mask']

        # Add the preprocess transform (normalization)
        if self.preprocess_transform:
            rgb, mask = self.apply_transforms(self.preprocess_transform, rgb, mask)

        # Add the augment transform (for training)
        if self.augment_transform:
            rgb, mask = self.apply_transforms(self.augment_transform, rgb, mask)

        # Convert to torch tensor
        rgb, mask = self.apply_transforms(DEFAULT, rgb, mask)

        return rgb, mask
