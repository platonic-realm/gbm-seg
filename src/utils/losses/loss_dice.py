"""
Author: Arash Fatehi
Date:   20.10.2022
File:   loss_dice.py
"""

# Python Imports

# Library Imports
import torch
import torch.nn.functional as Fn
from torch import nn
from torch.autograd import Variable

# Local Imports
from src.utils.misc import expand_as_one_hot


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, _weight=None, _normalization='sigmoid'):
        super().__init__()
        self.register_buffer('weight', _weight)

        assert _normalization in ['sigmoid', 'softmax', 'none']
        if _normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif _normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, _input, _target, _weight):
        raise NotImplementedError

    def forward(self, _input, _target):
        # get probabilities from logits
        _input = self.normalization(_input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(_input, _target, _weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """
    Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter
    can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit
    and will be normalized by the Sigmoid function.
    """

    def dice(self, _input, _target, _weight):
        return compute_per_channel_dice(_input, _target, _weight=self.weight)


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, _alpha, _beta):
        super().__init__()
        self.alpha = _alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = _beta
        self.dice = DiceLoss()

    def forward(self, _input, _target):
        return self.alpha * self.bce(_input, _target) \
               + self.beta * self.dice(_input, _target)


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, _input, _target, _weights):
        assert _target.size() == _weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(_input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW),
        # so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(_target,
                                   _c=_input.size()[1],
                                   _ignore_index=self.ignore_index)
        # expand weights
        _weights = _weights.unsqueeze(1)
        _weights = _weights.expand_as(_input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(
                    _input.size()[1]).float().to(_input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        _weights = class_weights * _weights

        # compute the losses
        result = -_weights * target * log_probabilities
        # average the losses
        return result.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
    WeightedCrossEntropyLoss (WCE)
    https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, _ignore_index=-1):
        super().__init__()
        self.ignore_index = _ignore_index

    def forward(self, _input, _target):
        weight = self._class_weights(_input)
        return Fn.cross_entropy(_input, _target, weight=weight,
                                ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(_input):
        # normalize the input first
        _input = Fn.softmax(_input, dim=1)
        flattened = flatten(_input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


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


def compute_per_channel_dice(_input, _target, _epsilon=1e-6, _weight=None):
    """
    Computes DiceCoefficient as defined in
    https://arxiv.org/abs/1606.04797 given a multi channel input and target.
    Assumes the input is a normalized probability,
    e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert _input.size() == _target.size(), \
           "'input' and 'target' must have the same shape"

    _input = flatten(_input)
    _target = flatten(_target)
    _target = _target.float()

    # compute per channel Dice Coefficient
    intersect = (_input * _target).sum(-1)
    if _weight is not None:
        intersect = _weight * intersect

    # here we can use standard dice (input + target).sum(-1) or
    # extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (_input * _input).sum(-1) + (_target * _target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=_epsilon))
