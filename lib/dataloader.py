from torch.utils.data import DataLoader, random_split
from lib.dataset import SatteliteImageDataset

class SatelliteImageDataLoader:
    train_dataset: SatteliteImageDataset
    val_dataset: SatteliteImageDataset
    test_dataset: SatteliteImageDataset

    def __init__(self, data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, preprocess_transform=None, transform=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.preprocess_transform = preprocess_transform
        self.augment_transform = transform

        # create the full dataset
        self.full_dataset = SatteliteImageDataset(data_dir, preprocess_transform)

        # calculate split sizes
        total_size = len(self.full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        # Make sure the split sizes add up to the total size
        assert train_size + val_size + test_size == total_size, "Split sizes do not add up to total size"

        # split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size]
        )

        # Add the augmentations to only train dataset
        self.train_dataset.augment_transform = self.augment_transform

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_all_loaders(self):
        return self.get_train_dataloader(), self.get_val_dataloader(), self.get_test_dataloader()
