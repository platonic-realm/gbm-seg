"""
Author: Arash Fatehi
Date:   07.09.2023
File:   loss_dice.py
"""

# Library Imports
import torch
import torch.nn.functional as Fn
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, _weights, _num_class=2, _epsilon=1e-6):
        super().__init__()
        self.register_buffer('weight', _weights)
        self.num_class = _num_class
        self.epsilon = _epsilon
        self.norm = nn.Softmax(dim=4)

    def forward(self, _input, _target):
        _target = Fn.one_hot(_target, self.num_class)
        _input = _input.permute(0, 2, 3, 4, 1)
        _input = self.norm(_input)
        return 1 - torch.mean(self.dice(_input, _target))

    def dice(self, _input, _target):
        intersect = (_input * _target).sum(-1)
        denominator = (_input * _input).sum(-1) + (_target * _target).sum(-1)
        return 2 * (intersect / denominator.clamp(min=self.epsilon))
