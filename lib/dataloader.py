import os
import random
import torch
from torch.utils.data import DataLoader, random_split

from lib.dataset import SatteliteImageDataset


class SatelliteImageDataLoader:
    train_dataset: SatteliteImageDataset
    val_dataset: SatteliteImageDataset
    test_dataset: SatteliteImageDataset

    def __init__(
        self,
        data_dir,
        batch_size=32,
        seed=None,
        preprocess_transform=None,
        transform=None,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.preprocess_transform = preprocess_transform
        self.augment_transform = transform
        self.seed = (
            seed if seed is not None else random.randint(0, 2**32 - 1)
        )  # Generate a random seed if not provided
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)

        # Check for train, val, and test directories
        assert os.path.exists(
            os.path.join(self.data_dir, "train")
        ), f"Train directory does not exist: {os.path.join(self.data_dir, 'train')}"
        assert os.path.exists(
            os.path.join(self.data_dir, "val")
        ), f"Val directory does not exist: {os.path.join(self.data_dir, 'val')}"
        assert os.path.exists(
            os.path.join(self.data_dir, "test")
        ), f"Test directory does not exist: {os.path.join(self.data_dir, 'test')}"

        # Create all the datasets
        #   Note: Add the augmentations to only the train dataset
        self.train_dataset = SatteliteImageDataset(
            os.path.join(self.data_dir, "train"),
            preprocess_transform=self.preprocess_transform,
            augment_transform=self.augment_transform,
        )
        self.val_dataset = SatteliteImageDataset(
            os.path.join(self.data_dir, "val"),
            preprocess_transform=self.preprocess_transform,
        )
        self.test_dataset = SatteliteImageDataset(
            os.path.join(self.data_dir, "test"),
            preprocess_transform=self.preprocess_transform,
        )

        # Total number of images
        self.size = (
            len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)
        )

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.gen,
        )

    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_all_loaders(self):
        return (
            self.get_train_dataloader(),
            self.get_val_dataloader(),
            self.get_test_dataloader(),
        )

    def __len__(self) -> int:
        return self.size
