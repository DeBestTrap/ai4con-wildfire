import torch

from lib.data import remove_filler_values
from lib.metrics.accuracy import pixel_accuracy
from lib.metrics.dice import dice_coefficient
from lib.metrics.iou import iou


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on data and compute metrics and loss.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): The DataLoader for the validation set.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to use for evaluation.

    Returns:
        Tuple: Average validation loss, IoU, Dice Coefficient, and Pixel Accuracy.
    """
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        iou_scores = []
        dice_scores = []
        pixel_accs = []

        for images, masks in loader:
            loss, main_outputs, masks = compute_loss_and_outputs(
                model, images, masks, criterion, device
            )
            val_loss += loss

            preds = post_process_predictions(main_outputs)
            clean_masks, clean_preds = clean_masks_and_predictions(masks, preds)

            masks_tensor = torch.stack(clean_masks)
            preds_tensor = torch.stack(clean_preds)

            # Calculate metrics
            iou_scores.extend(iou(masks_tensor, preds_tensor).tolist())
            dice_scores.extend(dice_coefficient(masks_tensor, preds_tensor).tolist())
            pixel_accs.extend(pixel_accuracy(masks_tensor, preds_tensor).tolist())

        # Calculate averages
        avg_loss = val_loss / len(loader)
        avg_iou = sum(iou_scores) / len(iou_scores)
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_pixel_acc = sum(pixel_accs) / len(pixel_accs)

        # NOTE: For debugging purposes
        # print()
        # print(f"Validation Loss: {avg_loss:.4f}")
        # print(f"Average IoU: {avg_iou:.4f}")
        # print(f"Average Dice Coefficient: {avg_dice:.4f}")
        # print(f"Average Pixel Accuracy: {avg_pixel_acc:.4f}")

        return avg_loss, avg_iou, avg_dice, avg_pixel_acc


def compute_loss_and_outputs(model, images, masks, criterion, device):
    """
    Perform a forward pass through the model and compute the loss.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        images (torch.Tensor): The input images.
        masks (torch.Tensor): The ground-truth masks.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to use.

    Returns:
        Tuple: Loss value, main model outputs, and processed masks.
    """
    images = images.to(device)
    masks = masks.to(device)
    masks = masks.where(masks < 2, torch.tensor(0, device=device)).float()

    outputs = model(images)
    main_outputs = outputs['out'].squeeze(1)
    aux_outputs = outputs['aux'].squeeze(1)

    main_loss = criterion(main_outputs, masks)
    aux_loss = criterion(aux_outputs, masks)
    loss = main_loss + 0.4 * aux_loss

    return loss.item(), main_outputs, masks


def post_process_predictions(main_outputs):
    """
    Apply sigmoid and threshold to generate binary predictions.

    Parameters:
        main_outputs (torch.Tensor): The main model outputs.

    Returns:
        torch.Tensor: Binary predictions.
    """
    return (torch.sigmoid(main_outputs) > 0.5).cpu()


def clean_masks_and_predictions(masks, preds):
    """
    Remove filler values from masks and predictions.

    Parameters:
        masks (torch.Tensor): The ground-truth masks.
        preds (torch.Tensor): The predicted masks.

    Returns:
        Tuple: Cleaned ground-truth masks and predictions as lists.
    """
    clean_masks = [remove_filler_values(mask) for mask in masks.cpu()]
    clean_preds = [remove_filler_values(pred) for pred in preds]
    return clean_masks, clean_preds
