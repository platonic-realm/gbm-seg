# Library Imports
import torch
import torch.nn as nn


def continuity_loss_diff(logits):
    """
    Computes the mean absolute difference between all pairs of adjacent depth slices in the logits.

    Args:
        logits (torch.Tensor): The model output with shape (batch_size, depth, height, width).

    Returns:
        torch.Tensor: The continuity loss value.
    """
    # Ensure logits has the expected shape
    assert logits.dim() == 5, "Logits should be a 5D tensor (batch_size, depth, height, width, logits)"

    # Compute the difference between adjacent depth slices
    # logits_current and logits_next both have shape (batch_size, depth-1, height, width)
    logits_current = logits[:, :-1, :, :, :]  # All slices except the last one
    logits_next = logits[:, 1:, :, :, :]      # All slices except the first one

    # Compute the absolute differences
    differences = torch.abs(logits_current - logits_next)  # Shape: (batch_size, depth-1, height, width)

    # Aggregate the differences
    # You can choose to sum or mean over specific dimensions
    # Here, we take the mean over all dimensions to get a scalar loss value
    cont_loss = differences.mean()

    return cont_loss


class UncertainityLoss(nn.Module):
    def __init__(self, _cross_entropy):
        super().__init__()
        self.log_sigma_ce = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cont = nn.Parameter(torch.tensor(0.0))
        self.cross_entropy = _cross_entropy

    def forward(self, _logits, _labels):
        sigma_ce = torch.exp(self.log_sigma_ce)
        sigma_cont = torch.exp(self.log_sigma_cont)

        ce_loss = self.cross_entropy(_logits, _labels)
        cont_loss = continuity_loss_diff(_logits)

        total_loss = (1 / (2 * sigma_ce**2)) * ce_loss + torch.log(sigma_ce) + \
                     (1 / (2 * sigma_cont**2)) * cont_loss + torch.log(sigma_cont)
        return total_loss
