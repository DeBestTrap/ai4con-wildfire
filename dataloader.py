from torch.utils.data import DataLoader, random_split
from dataset import SatteliteImageDataset

class SatelliteImageDataLoader:
    def __init__(self, data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, transform=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

        # create the full dataset
        self.full_dataset = SatteliteImageDataset(data_dir, transform)

        # calculate split sizes
        total_size = len(self.full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        # split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size]
        )

        def get_train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        def get_val_dataloader(self):
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        def get_test_dataloader(self):
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
