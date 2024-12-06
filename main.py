# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
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

# %%


data_dir = 'data'
batch_size = 32
seed = 0

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
loader = SatelliteImageDataLoader(data_dir, batch_size=batch_size, seed=seed, preprocess_transform=preprocess_transform, transform=augment_transform)
print(f"Number of images in full dataset: {len(loader)}")

train_loader, val_loader, test_loader = loader.get_all_loaders()
print(f"Number of images in train dataset: {len(train_loader.dataset)}")
print(f"Number of images in val dataset: {len(val_loader.dataset)}")
print(f"Number of images in test dataset: {len(test_loader.dataset)}")

# for batch in train_loader:
#     rgb, mask = batch 
#     rgb = rgb[0]
#     mask = mask[0]
#     if 1 in mask:
#         print("Found mask")
#         print(rgb.shape, mask.shape)
#         plot_preprocessed_rgb(rgb, normalized=True)
#         # plot_mask(mask)
#         # plot_burned_area_mask(mask)
#         # plot_highlighted_rgb_and_mask(rgb, mask, only_burned=True)
#         plot_highlighted_rgb_and_mask_preprocessed(rgb, mask, normalized=True, only_burned=False)
#         break
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


    # Modify the classifier to output 2 classes
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    # (Optional) Modify the auxiliary classifier if used
    model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

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


model = ModelWrapper(deeplabv3_init(), output_transform=output_transform)
from lib.train import train_multiple_epochs
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
train_multiple_epochs(model, train_loader, criterion, optimizer, device, num_epochs, viz)

# model.eval()

# with torch.no_grad():
#     for batch in train_loader:
#         rgb, mask = batch 
#         rgb = rgb[0]
#         mask = mask[0]
#         if 1 in mask:
#             print("Found mask")
#             print(rgb.shape, mask.shape)
#             # plot_preprocessed_rgb(rgb, normalized=True)
#             # plot_mask(mask)
#             plot_highlighted_rgb_and_mask_preprocessed(rgb, mask, normalized=True, only_burned=False)
#             batch_rgb = rgb.unsqueeze(0)
#             print(batch_rgb.shape)
#             logits = model(batch_rgb)
#             batch_pred_mask = logits.argmax(dim=1)
#             # plot_highlighted_rgb_and_mask_preprocessed(rgb, batch_pred_mask[0], normalized=True, only_burned=False)
#             plot_mask(batch_pred_mask[0])
#             break


# %%
