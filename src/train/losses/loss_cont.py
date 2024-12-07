# Library Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


def continuity_loss_diff(logits):
    """
    Computes the mean absolute difference between all pairs of adjacent depth slices in the logits.

    Args:
        logits (torch.Tensor): The model output with shape (batch_size, depth, height, width, logits)

    Returns:
        torch.Tensor: The continuity loss value.
    """
    # Ensure logits has the expected shape
    assert logits.dim() == 5, "Logits should be a 5D tensor (batch_size, depth, height, width, logits)"

    # Compute the difference between adjacent depth slices
    logits_current = logits[:, :-1, :, :, :]  # All slices except the last one
    logits_next = logits[:, 1:, :, :, :]      # All slices except the first one

    # Compute the absolute differences
    differences = torch.abs(logits_current - logits_next)  # Shape: (batch_size, depth-1, height, width, logits)

    # Aggregate the differences
    cont_loss = differences.mean()

    return cont_loss


class ContLoss(nn.Module):
    def __init__(self, cross_entropy_loss, min_alpha=0.8, min_beta=0.05, reg_lambda=0.01):
        super().__init__()
        self.w_alpha = nn.Parameter(torch.tensor(0.0))
        self.w_beta = nn.Parameter(torch.tensor(0.0))
        self.cross_entropy_loss = cross_entropy_loss
        self.min_alpha = min_alpha
        self.min_beta = min_beta
        self.reg_lambda = reg_lambda

        if self.min_alpha + self.min_beta > 1.0:
            raise ValueError("The sum of min_alpha and min_beta must be less than or equal to 1.")

    def forward(self, logits, labels):
        ce_loss = self.cross_entropy_loss(logits, labels)
        cont_loss = continuity_loss_diff(logits)

        variable_weight = 1.0 - self.min_alpha - self.min_beta

        if variable_weight < 0.0:
            raise ValueError("The sum of min_alpha and min_beta must be less than or equal to 1.")

        # Apply sigmoid to ensure weights are between 0 and 1
        scaled_alpha = torch.sigmoid(self.w_alpha)
        scaled_beta = torch.sigmoid(self.w_beta)

        # Normalize the scaled weights
        weight_sum = scaled_alpha + scaled_beta + 1e-6  # Avoid division by zero
        scaled_alpha_norm = scaled_alpha / weight_sum
        scaled_beta_norm = scaled_beta / weight_sum

        # Adjust weights with minimums
        alpha = self.min_alpha + variable_weight * scaled_alpha_norm
        beta = self.min_beta + variable_weight * scaled_beta_norm

        total_loss = alpha * ce_loss + beta * cont_loss

        reg_loss = self.reg_lambda * (self.w_alpha.pow(2) + self.w_beta.pow(2))
        total_loss += reg_loss

        return total_loss
