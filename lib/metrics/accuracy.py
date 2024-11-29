import torch
from torchmetrics.classification import Accuracy

def pixel_accuracy(ground_truth, predicted):
    """
    Compute Pixel Accuracy for a batch.

    Args:
        ground_truth (torch.Tensor): Ground truth masks,
            shape [batch_size, height, width].
        predicted (torch.Tensor): Predicted masks,
            shape [batch_size, height, width].

    Returns:
        torch.Tensor: Pixel accuracy for each sample in the batch, shape [batch_size].
    """
    # Ensure tensors are binary
    ground_truth = ground_truth.int()
    predicted = predicted.int()

    batch_size = ground_truth.shape[0]
    accuracies = torch.zeros(batch_size, device=ground_truth.device)

    # Initialize accuracy metric for binary classification
    accuracy_metric = Accuracy(task="binary").to(ground_truth.device)

    # Loop through each sample in the batch
    for i in range(batch_size):
        # Compute accuracy for the current sample
        accuracies[i] = accuracy_metric(predicted[i].unsqueeze(0), ground_truth[i].unsqueeze(0))

    return accuracies
