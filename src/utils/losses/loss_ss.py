"""
Author: Arash Fatehi
Date:   21.02.2022
File:   loss_ss.py
"""

# Python Imports
from enum import Enum

# Library Imports
import torch
from torch import nn

# Local Imports


class SSMode(Enum):
    Supervised = 0
    Unsupervised = 1


class SelfSupervisedLoss(nn.Module):
    def __init__(self,
                 _no_of_epochs: int,
                 _weight: list):

        super().__init__()
        self.weight = _weight
        self.no_of_epochs = _no_of_epochs

        self.mode = SSMode.Unsupervised

        self.unsupervised_loss = nn.MSELoss()
        self.supvised_loss = nn.CrossEntropyLoss(weight=_weight)

    def forward(self,
                _epoch_no: int,
                _predicted_segmentation,
                _predicted_interpolation,
                _truth_segmentation,
                _truth_interpolation):

        alpha = 1 - _epoch_no/self.no_of_epochs
        alpha = alpha*10

        unsupervised_part = \
            torch.mul(alpha,
                      self.unsupervised_loss(_predicted_interpolation,
                                             _truth_interpolation))
        if self.mode == SSMode.Unsupervised:
            return unsupervised_part

        return torch.add(unsupervised_part,
                         self.supvised_loss(_predicted_segmentation,
                                            _truth_segmentation))

    def supervised(self):
        self.mode = SSMode.Supervised

    def unsupervised(self):
        self.mode = SSMode.Unsupervised
