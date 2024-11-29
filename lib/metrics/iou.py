from torchmetrics.functional import jaccard_index
import torch

def iou(ground_truth, predicted):
    """
    Compute IoU for a batch.

    Args:
        ground_truth (torch.Tensor): Ground truth masks,
            shape [batch_size, height, width].
        predicted (torch.Tensor): Predicted masks,
            shape [batch_size, height, width].

    Returns:
        torch.Tensor: IoU for each sample in the batch, shape [batch_size].
    """
    ground_truth = ground_truth.int()
    predicted = predicted.int()

    batch_size = ground_truth.shape[0]
    iou_scores = torch.zeros(batch_size, device=ground_truth.device)

    # Compute IoU for each sample in the batch independently
    for i in range(batch_size):
        iou_scores[i] = jaccard_index(
            predicted[i].unsqueeze(0),
            ground_truth[i].unsqueeze(0),
            num_classes=2,
            task="binary"
        )

    return iou_scores
