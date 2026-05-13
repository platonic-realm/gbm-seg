"""Sanity checks for Dice/IoU losses at the extremes."""

import torch

from src.train.losses.loss_dice import DiceLoss
from src.train.losses.loss_iou import IoULoss


def _binary_logits(target, confident=20.0):
    """Build logits that the softmax will collapse to one-hot `target`."""
    B, D, H, W = target.shape
    logits = torch.zeros(B, 2, D, H, W)
    logits[:, 0, :, :, :] = (1 - target) * confident
    logits[:, 1, :, :, :] = target * confident
    return logits


def test_dice_perfect_overlap():
    target = torch.zeros(1, 4, 8, 8)
    target[:, :, :4, :4] = 1
    weights = torch.tensor([1.0, 1.0])
    loss = DiceLoss(_weights=weights)(_binary_logits(target), target.long())
    assert loss.item() < 0.05


def test_iou_perfect_overlap():
    target = torch.zeros(1, 4, 8, 8)
    target[:, :, :4, :4] = 1
    weights = torch.tensor([1.0, 1.0])
    loss = IoULoss(_weights=weights)(_binary_logits(target), target.long())
    assert loss.item() < 0.05


def test_dice_disjoint_predictions():
    target = torch.zeros(1, 4, 8, 8)
    target[:, :, :4, :4] = 1
    inverted_logits = _binary_logits(1 - target)
    weights = torch.tensor([1.0, 1.0])
    loss = DiceLoss(_weights=weights)(inverted_logits, target.long())
    assert loss.item() > 0.9
