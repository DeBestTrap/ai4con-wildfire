import torch

from lib.data import remove_filler_values
from lib.metrics.accuracy import pixel_accuracy
from lib.metrics.dice import dice_coefficient
from lib.metrics.iou import iou


def evaluate(model, loader, criterion, threshold, device):
    """
    Evaluate the model on data and compute metrics and loss.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): The DataLoader for the validation set.
        criterion (torch.nn.Module): The loss function.
        threshold (float): The threshold for post-processing predictions.
        device (torch.device): The device to use for evaluation.

    Returns:
        Tuple: Average validation loss, IoU, Dice Coefficient, Pixel Accuracy, and TP, TN, FP, FN.
    """
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        iou_scores = []
        dice_scores = []
        pixel_accs = []
        tp_tn_fp_fn_total = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        for images, masks in loader:
            loss, main_outputs, masks = compute_loss_and_outputs(
                model, images, masks, criterion, device
            )
            val_loss += loss

            preds = post_process_predictions(main_outputs, threshold)
            clean_masks, clean_preds = clean_masks_and_predictions(masks, preds)

            masks_tensor = torch.stack(clean_masks)
            preds_tensor = torch.stack(clean_preds)

            # Calculate metrics
            iou_scores.extend(iou(masks_tensor, preds_tensor).tolist())
            dice_scores.extend(dice_coefficient(masks_tensor, preds_tensor).tolist())
            pixel_accs.extend(pixel_accuracy(masks_tensor, preds_tensor).tolist())

            # Calculate TP, TN, FP, FN
            tp_tn_fp_fn = calculate_tp_tn_fp_fn(preds_tensor, masks_tensor)
            for key, value in tp_tn_fp_fn.items():
                tp_tn_fp_fn_total[key] += value

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

        return avg_loss, avg_iou, avg_dice, avg_pixel_acc, tp_tn_fp_fn_total

def calculate_tp_tn_fp_fn(pred_mask, true_mask):
    """
    Calculate TP, TN, FP, FN pixels from predicted and true masks.

    Parameters:
        pred_mask (torch.Tensor): Predicted binary mask.
        true_mask (torch.Tensor): Ground truth binary mask.

    Returns:
        dict: Dictionary containing TP, TN, FP, FN counts.
    """
    if pred_mask.shape != true_mask.shape:
        raise ValueError("Predicted and true masks must have the same shape.")

    # Calculate components
    TP = torch.sum((pred_mask == 1) & (true_mask == 1)).item()
    TN = torch.sum((pred_mask == 0) & (true_mask == 0)).item()
    FP = torch.sum((pred_mask == 1) & (true_mask == 0)).item()
    FN = torch.sum((pred_mask == 0) & (true_mask == 1)).item()

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

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
    main_outputs = outputs["out"].squeeze(1)
    aux_outputs = outputs["aux"].squeeze(1)

    main_loss = criterion(main_outputs, masks)
    aux_loss = criterion(aux_outputs, masks)
    loss = main_loss + 0.4 * aux_loss

    return loss.item(), main_outputs, masks


def post_process_predictions(main_outputs, threshold):
    """
    Apply sigmoid and threshold to generate binary predictions.

    Parameters:
        main_outputs (torch.Tensor): The main model outputs.
        threshold (float): The threshold value.

    Returns:
        torch.Tensor: Binary predictions.
    """
    return (torch.sigmoid(main_outputs) > threshold).cpu()


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
