import torch

from lib.metrics.accuracy import pixel_accuracy
from lib.metrics.dice import dice_coefficient
from lib.metrics.iou import iou


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on data and compute metrics and loss

    TODO add metrics
    """
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
