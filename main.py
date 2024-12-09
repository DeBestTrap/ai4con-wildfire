# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from collections import OrderedDict
from visdom import Visdom
from tqdm import tqdm
import time
import shutil
import json
import argparse

from lib.dataloader import SatelliteImageDataLoader
from lib.dataset import SatteliteImageDataset
from lib.debug import *
from lib.model_wrapper import ModelWrapper
from lib.data import seperate_visible_and_infrared, InverseNormalize
from lib.config import load_config
from lib.train import train_multiple_epochs
from lib.evaluate import evaluate

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=False)
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

# Initialize Visdom instance
viz = Visdom(env="ai4con-wildfire")
assert viz.check_connection(), "Visdom server is not running!"

# Create experiment directory
current_time = time.strftime("%Y%m%d-%H%M%S")
experiment_name = f"{current_time}_{args.name}"
experiment_dir = os.path.join("experiments", experiment_name)
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# Load experiment configuration
experiment_config, augment_transform = load_config(config_path=args.config)
shutil.copyfile(args.config, os.path.join(experiment_dir, "config.yml"))

# Get mean and std of each channel from file for normalization preprocessing transform
with open(os.path.join(experiment_config['loader_params']['data_dir'], 'mean_std.json'), 'r') as f:
    mean_std = json.load(f)
preprocess_transform = A.Compose([
    A.Normalize(mean=mean_std['mean'], std=mean_std['std'], max_pixel_value=1.0),
])

# Create inverse normalization transform for visualization purposes later
inverse_normalize_transform = A.Compose([
    InverseNormalize(mean=mean_std['mean'], std=mean_std['std']),
])

loader = SatelliteImageDataLoader(**experiment_config['loader_params'], preprocess_transform=preprocess_transform, transform=augment_transform)
print(f"Total number of images: {len(loader)}")

train_loader, val_loader, test_loader = loader.get_all_loaders()
print(f"Number of images in train dataset: {len(train_loader.dataset)}")
print(f"Number of images in val dataset: {len(val_loader.dataset)}")
print(f"Number of images in test dataset: {len(test_loader.dataset)}")

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

def train(max_epochs: int, save_on_metric: str, viz: Visdom) -> None:
    train_multiple_epochs(model, train_loader, val_loader, criterion, optimizer,
                          device, max_epochs, save_on_metric, experiment_dir, viz)

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

def eval_model(save_on_metric: str, loader: DataLoader, name: str, **kwargs):
    model.load_state_dict(torch.load(os.path.join(experiment_dir, f"model_{save_on_metric}.pt")))
    model.eval()
    with torch.no_grad():
        val_loss, iou_score, dice_score, pixel_acc = evaluate(model, loader, criterion, device)
    
    # dump results json
    results = {
        "val_loss": val_loss,
        "iou_score": iou_score,
        "dice_score": dice_score,
        "pixel_acc": pixel_acc
    }
    with open(os.path.join(experiment_dir, f"results_{name}.json"), "w") as f:
        json.dump(results, f)


def get_criterion(name: str, **kwargs) -> torch.nn.Module:
    if name == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss(**kwargs)
    elif name == "FocalLoss":
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown criterion: {name}")

def get_optimizer(model: torch.nn.Module, name: str, **kwargs) -> torch.optim.Optimizer:
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = ModelWrapper(deeplabv3_init(), output_transform=output_transform)
model = deeplabv3_init()
model.to(device)
criterion = get_criterion(**experiment_config['criterion'])
optimizer = get_optimizer(model, **experiment_config['optimizer'])

train(**experiment_config['train'], viz=viz)
eval_model(**experiment_config['train'], loader=val_loader, name="val")
eval_model(**experiment_config['train'], loader=test_loader, name="test")
# show_an_output()

# %%
