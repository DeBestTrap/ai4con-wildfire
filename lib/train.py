import time
import torch
from torch.utils.data import DataLoader
from visdom import Visdom
from tqdm import tqdm

from lib.debug import *
from lib.evaluate import evaluate


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch

    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The DataLoader for the training set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        device (torch.device): The device to use for training.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images: torch.Tensor = images.to(device)
        masks: torch.Tensor = masks.to(device)
        masks = masks.where(masks < 2, 0).float()

        optimizer.zero_grad()
        main_outputs = model(images)['out'].squeeze(1)
        aux_outputs = model(images)['aux'].squeeze(1)
        # outputs = model(images).squeeze(1)

        # loss = criterion(outputs, masks)
        main_loss = criterion(main_outputs, masks)
        aux_loss = criterion(aux_outputs, masks)
        loss = main_loss + 0.4 * aux_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def train_multiple_epochs(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    visdom_instance: Visdom = None,
) -> None:
    """
    Train the model for multiple epochs

    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): The DataLoader for the training set.
        val_loader(DataLoader): The DataLoader for the validation set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        device (torch.device): The device to use for training.
        num_epochs (int): The number of epochs to train for.
        visdom_instance (Visdom): The Visdom instance to use for live visualization.
    """

    # Book-keeping
    training_losses = []
    val_losses = []
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []

    train_win = visdom_instance.line(Y=[0], X=[0], opts=dict(title="Training Loss"))
    val_loss_win = visdom_instance.line(Y=[0], X=[0], opts=dict(title="Validation Loss"))
    metrics_win = visdom_instance.line(
        Y=[[0, 0, 0]],
        X=[0],
        opts=dict(
            title="Validation Metrics",
            legend=["IoU", "Dice", "Pixel Accuracy"],
            xlabel="Epoch",
            ylabel="Metric Value",
        ),
    )

    for epoch in tqdm(range(num_epochs)):
        # Training
        avg_epoch_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        val_results = evaluate(model, val_loader, criterion, device)
        avg_val_loss, avg_iou, avg_dice, avg_pixel_acc = val_results

        # Book-keeping
        training_losses.append(avg_epoch_loss)
        val_losses.append(avg_val_loss)
        iou_scores.append(avg_iou)
        dice_scores.append(avg_dice)
        pixel_accuracies.append(avg_pixel_acc)

        # Update Visdom
        visdom_instance.line(
            Y=[avg_epoch_loss], X=[epoch], win=train_win, update="append"
        )
        visdom_instance.line(
            Y=[avg_val_loss], X=[epoch], win=val_loss_win, update="append"
        )
        visdom_instance.line(
            Y=[[avg_iou, avg_dice, avg_pixel_acc]],
            X=[epoch],
            win=metrics_win,
            update="append",
        )

    # Save the model
    torch.save(model.state_dict(), "model.pt")

    print("Training complete!")
