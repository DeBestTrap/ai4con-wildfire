import torch
from torchmetrics.segmentation import DiceScore

def dice_coefficient(ground_truth, predicted):
    """
    Compute Dice coefficient for a batch.

    Args:
        ground_truth (torch.Tensor): Ground truth masks,
            shape [batch_size, height, width].
        predicted (torch.Tensor): Predicted masks,
            shape [batch_size, height, width].

    Returns:
        torch.Tensor: Dice coefficient for each sample in the batch, shape [batch_size].
    """
    ground_truth = ground_truth.int()
    predicted = predicted.int()

    batch_size = ground_truth.shape[0]
    dice_scores = torch.zeros(batch_size, device=ground_truth.device)

    for i in range(batch_size):
        # Prepare data for a single sample
        gt = ground_truth[i].unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]
        pred = predicted[i].unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]

        # Compute Dice score for this sample
        dice_metric = DiceScore(num_classes=2, average="micro").to(ground_truth.device)
        sample_dice_score = dice_metric(pred, gt)  # Single value for binary segmentation
        dice_scores[i] = sample_dice_score

    return dice_scores
