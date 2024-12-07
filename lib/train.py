import time
import torch
from torch.utils.data import DataLoader
from visdom import Visdom
from tqdm import tqdm

from lib.debug import *


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
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        device (torch.device): The device to use for training.
        num_epochs (int): The number of epochs to train for.
        visdom_instance (Visdom): The Visdom instance to use for live visualization.
    """

    # Book-keeping
    training_losses = []
    line_win = visdom_instance.line(Y=[0], X=[0], opts=dict(title="Training Loss"))

    for epoch in tqdm(range(num_epochs)):
        # Training
        avg_epoch_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation

        # Book-keeping
        training_losses.append(avg_epoch_loss)
        visdom_instance.line(
            Y=[avg_epoch_loss], X=[epoch], win=line_win, update="append"
        )

    # Save the model
    torch.save(model.state_dict(), "model.pt")

    print("Training complete!")
