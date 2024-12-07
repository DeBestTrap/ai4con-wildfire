# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torch
import torch.nn as nn
import albumentations as A
from collections import OrderedDict
from visdom import Visdom
from tqdm import tqdm

from lib.dataloader import SatelliteImageDataLoader
from lib.dataset import SatteliteImageDataset
from lib.debug import *
from lib.model_wrapper import ModelWrapper

# Initialize Visdom instance
viz = Visdom()
assert viz.check_connection(), "Visdom server is not running!"

data_dir = 'data_full_burned'
batch_size = 10
seed = None

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

loader = SatelliteImageDataLoader(data_dir, batch_size=batch_size, seed=seed, preprocess_transform=preprocess_transform, transform=augment_transform)
print(f"Number of images in full dataset: {len(loader)}")

train_loader, val_loader, test_loader = loader.get_all_loaders()
print(f"Number of images in train dataset: {len(train_loader.dataset)}")
print(f"Number of images in val dataset: {len(val_loader.dataset)}")
print(f"Number of images in test dataset: {len(test_loader.dataset)}")

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

    threshold = 0.2
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
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    num_epochs = 15
    train_multiple_epochs(model, train_loader, criterion, optimizer, device, num_epochs, viz)

def show_an_output():
    # temp function
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            rgb, mask = batch 
            rgb = rgb[0]
            mask = mask[0]
            if 1 in mask:
                print("Found mask")
                # plot_preprocessed_rgb(rgb, normalized=True)
                # plot_mask(mask)
                plot_highlighted_rgb_and_mask_preprocessed(rgb, mask, normalized=True, only_burned=False)
                batch_rgb = rgb.unsqueeze(0).to(device)
                batch_pred_mask = predict_mask(model, batch_rgb)
                plot_highlighted_rgb_and_mask_preprocessed(rgb, batch_pred_mask[0], normalized=True, only_burned=False)
                # plot_mask(batch_pred_mask[0])
                return

train()
# show_an_output()