# Library Imports
import torch
import torch.nn as nn


def continuity_loss_diff(logits):
    # logits shape: (B, C, D, H, W) — softmax over the class dim, diff along depth.
    assert logits.dim() == 5, "Logits should be a 5D tensor (B, C, D, H, W)"

    probs = torch.softmax(logits, dim=1)

    probs_current = probs[:, :, :-1, :, :]
    probs_next = probs[:, :, 1:, :, :]

    return torch.abs(probs_current - probs_next).mean()


class ContLoss(nn.Module):
    def __init__(self,
                 cross_entropy_loss,
                 min_alpha=0.6,
                 min_beta=0.3,
                 reg_lambda=0.01):

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
