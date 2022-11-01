"""
Author: Arash Fatehi
Date:   20.10.2022
File:   main.py
"""

# Python Imprts
import sys
import logging

# Library Imports
import torch
from torch.utils.data import DataLoader

# Local Imports
from src.models.unet3d_me import Unet3DME
from src.models.losses import DiceLoss
from src.configs import LOG_LEVEL
from src.utils.datasets import TrainingDataset
from src.trainer import train_3d_unet

if __name__ == '__main__':

    logging.basicConfig(
                    level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("testing.log"),
                              logging.StreamHandler(sys.stdout)])

    training_dataset = TrainingDataset(
        _source_directory='/data/afatehi/data/GBM-Train-DS',
        _sample_dimension=(12, 128, 128),
        _pixel_per_step=(1, 16, 16)
    )

    training_loader = DataLoader(training_dataset,
                                 batch_size=5,
                                 shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_3d = Unet3DME(1, _feature_maps=(128, 256, 512, 1024))
    optimizer = torch.optim.SGD(unet_3d.parameters(), lr=0.01, momentum=0.9)
    loss_function = DiceLoss()

    train_3d_unet(_model=unet_3d,
                  _dataloader=training_loader,
                  _device=device,
                  _optimizer=optimizer,
                  _loss_function=loss_function)
