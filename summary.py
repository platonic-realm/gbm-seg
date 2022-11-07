"""
Author: Arash Fatehi
Date:   06.11.2022
File:   summary.py
"""

# Library Imprts
from torchinfo import summary

# Local Imports
from src.models.unet3d import Unet3D

if __name__ == '__main__':
    model = Unet3D(_input_channels=3,
                   _feature_maps=(64, 128, 256, 512))

    summary(model, input_size=(4, 3, 12, 256, 256))
