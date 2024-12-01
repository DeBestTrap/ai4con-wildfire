import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from lib.data import remove_filler_values, rgb_float_to_uint8

class SatteliteImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('npz')]
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        rgb = data['rgb']
        mask = data['mask']

        # Apply preprocessing
        rgb = rgb_float_to_uint8(rgb)
        mask = remove_filler_values(mask)

        # convert to pytorch tensors
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        if self.transform:
            rgb = self.transform(rgb)
            mask = self.transform(mask)
        return rgb, mask
