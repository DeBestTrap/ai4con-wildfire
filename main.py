# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import torch
import torch.nn as nn
import albumentations as A
from collections import OrderedDict
from visdom import Visdom
from tqdm import tqdm
import json

from lib.dataloader import SatelliteImageDataLoader
from lib.dataset import SatteliteImageDataset
from lib.debug import *
from lib.model_wrapper import ModelWrapper
from lib.data import seperate_visible_and_infrared, InverseNormalize

# Initialize Visdom instance
viz = Visdom()
assert viz.check_connection(), "Visdom server is not running!"

# Parameters
data_dir = './data_infrared_split'
batch_size = 32
seed = None
augment_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
])


# Get mean and std of each channel from file for normalization preprocessing transform
with open(os.path.join(data_dir, 'mean_std.json'), 'r') as f:
    mean_std = json.load(f)
preprocess_transform = A.Compose([
    A.Normalize(mean=mean_std['mean'], std=mean_std['std'], max_pixel_value=1.0),
])
inverse_normalize_transform = A.Compose([
    InverseNormalize(mean=mean_std['mean'], std=mean_std['std']),
])

loader = SatelliteImageDataLoader(data_dir, batch_size=batch_size, seed=seed, preprocess_transform=preprocess_transform, transform=augment_transform)
print(f"Total number of images: {len(loader)}")

train_loader, val_loader, test_loader = loader.get_all_loaders()
print(f"Number of images in train dataset: {len(train_loader.dataset)}")
print(f"Number of images in val dataset: {len(val_loader.dataset)}")
print(f"Number of images in test dataset: {len(test_loader.dataset)}")
# for image, mask in train_loader:
#     # plot_highlighted_rgb_and_mask_preprocessed
#     # print(torch.max(image[0].reshape(6, -1), axis=1).values, torch.min(image[0].reshape(6, -1), axis=1).values)
#     image = unprocess_image(image[0], inverse_normalize_transform)
#     # print(np.max(image.reshape(6, -1), axis=1), np.min(image.reshape(6, -1), axis=1))
#     rgb, infrared = seperate_visible_and_infrared(image)
#     plot_rgb(rgb)
#     plot_rgb(infrared)
#     plot_mask(mask[0])
#     break

# %%

def deeplabv3_init() -> torch.nn.Module:
    """
    Initialize the DeepLabV3 model
    """
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
    )
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)

   # Modify the input to accept 6 channels
    model.backbone.conv1 = torch.nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Modify the classifier to output binary class
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    # (Optional) Modify the auxiliary classifier if used
    model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    return model


def output_transform(raw_output: OrderedDict) -> torch.Tensor:
    """
    Transform the raw output of the model to a standard format

    Parameters:
        raw_output (OrderedDict): The raw output from the model.

    Returns:
        A torch tensor of shape [batch_size, num_classes, height, width]
    """
    output = raw_output["out"]
    return output

def predict_mask(model: torch.nn.Module, rgb: torch.Tensor) -> torch.Tensor:
    '''
    Takes in a batch of rgb images
        Size: [batch_size, 3, height, width]
        or [3, height, width] (will be resized to [1, 3, height, width])

    Returns a batch of predicted masks of size [batch_size, height, width]
    '''
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)

    threshold = 0.5
    logits = model(rgb)
    logits = logits["out"]

    batch_pred_mask = logits.sigmoid().cpu().squeeze(1)
    print(torch.max(batch_pred_mask), torch.min(batch_pred_mask))
    batch_pred_mask = (batch_pred_mask > threshold).float()
    print(torch.max(batch_pred_mask), torch.min(batch_pred_mask))

    return batch_pred_mask

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of correct prediction
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = ModelWrapper(deeplabv3_init(), output_transform=output_transform)
model = deeplabv3_init()
model.to(device)

def train():
    # temp function
    from lib.train import train_multiple_epochs
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    num_epochs = 30
    train_multiple_epochs(model, train_loader, val_loader,criterion, optimizer,
                          device, num_epochs, viz)

def show_an_output():
    # temp function
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            image, mask = batch 
            image = image[0]
            mask = mask[0]
            rgb, infrared = seperate_visible_and_infrared(unprocess_image(image, inverse_normalize_transform))
            if 1 in mask:
                print("Found mask")
                plot_rgb(rgb)
                plot_mask(mask)
                batch_image = image.unsqueeze(0).to(device)
                batch_pred_mask = predict_mask(model, batch_image)
                plot_mask(batch_pred_mask[0])
                return

# train()
show_an_output()
