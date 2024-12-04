# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()

# %%
from lib.dataloader import SatelliteImageDataLoader
from lib.dataset import SatteliteImageDataset
from lib.debug import *
import albumentations as A

data_dir = 'data'
batch_size = 32
preprocess_transform = None
augment_transform = None

preprocess_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
augment_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
])

# dataset = SatteliteImageDataset(data_dir, preprocess_transform, augment_transform)
# for i in range(len(dataset)):
    # rgb, mask = dataset[i]
loader = SatelliteImageDataLoader(data_dir, batch_size=batch_size, preprocess_transform=preprocess_transform, transform=augment_transform)
train_loader, val_loader, test_loader = loader.get_all_loaders()
for batch in train_loader:
    rgb, mask = batch 
    rgb = rgb[0]
    mask = mask[0]
    if 1 in mask:
        print("Found mask")
        print(rgb.shape, mask.shape)
        plot_preprocessed_rgb(rgb, normalized=True)
        # plot_mask(mask)
        # plot_burned_area_mask(mask)
        # plot_highlighted_rgb_and_mask(rgb, mask, only_burned=True)
        plot_highlighted_rgb_and_mask_preprocessed(rgb, mask, normalized=True, only_burned=False)
        break