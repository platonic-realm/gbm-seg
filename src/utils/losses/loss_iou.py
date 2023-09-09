"""
Author: Arash Fatehi
Date:   08.09.2023
File:   loss_iou.py
"""

# Python Imports

# Library Imports
import torch
import torch.nn.functional as Fn
from torch import nn


class _AbstractIoULoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, _weights=None, _normalization='softmax'):
        super().__init__()
        self.register_buffer('weight', _weights)

        assert _normalization in ['sigmoid', 'softmax', 'none']
        if _normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif _normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def iou(self, _input, _target, _weight):
        raise NotImplementedError

    def forward(self, _input, _target):
        # get probabilities from logits
        _input = self.normalization(_input)

        # compute per channel Dice coefficient
        per_channel_iou = self.iou(_input, _target, _weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_iou)


class IoULoss(_AbstractIoULoss):
    def iou(self, _input, _target, _weight):
        return compute_per_channel_iou(_input, _target, _weight=self.weight)


def flatten(_tensor):
    """
    Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """

    # number of channels
    channels = _tensor.size(1)

    # new axis order
    axis_order = (1, 0) + tuple(range(2, _tensor.dim()))

    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = _tensor.permute(axis_order)

    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(channels, -1)


def compute_per_channel_iou(_input, _target, _epsilon=1e-6, _weight=None):
    _target = Fn.one_hot(_target, 2)
    _target = _target.permute(0, 4, 1, 2, 3)
    # input and target shapes must match
    assert _input.size() == _target.size(), \
           "'input' and 'target' must have the same shape"

    _input = flatten(_input)
    _target = flatten(_target)
    _target = _target.float()

    # compute per channel Dice Coefficient
    intersect = (_input * _target).sum(-1)
    union = (_input * _input).sum(-1) + (_target * _target).sum(-1) - intersect
    if _weight is not None:
        intersect = _weight * intersect

    return intersect / union.clamp(min=_epsilon)
