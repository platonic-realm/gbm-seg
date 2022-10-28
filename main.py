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
from torch import nn
from torch.utils.data import DataLoader

# Local Imports
from src.models.unet3d_me import Unet3DME
from src.configs import LOG_LEVEL
from src.utils.datasets import TrainingDataset

if __name__ == '__main__':

    logging.basicConfig(
                    level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("testing.log"),
                              logging.StreamHandler(sys.stdout)])

    training_dataset = TrainingDataset(
        _source_directory='/data/afatehi/GBM-Train-DS',
        _sample_dimension=(12, 128, 128),
        _pixel_per_step=(1, 16, 16)
    )

    training_loader = DataLoader(training_dataset,
                                 batch_size=12,
                                 shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_3d = Unet3DME(1, _feature_maps=(64, 128, 256, 512))
    unet_3d = nn.DataParallel(unet_3d)
    unet_3d.to(device)

    logging.info("Using device: %s", device)

    optimizer = torch.optim.SGD(unet_3d.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.MSELoss()

    for index, data in enumerate(training_loader):
        nephrin = data['nephrin'].to(device)
        wga = data['wga'].to(device)
        collagen4 = data['collagen4'].to(device)
        labels = data['labels'].to(device)
        offests = data['offsets']

        optimizer.zero_grad()

        outputs = unet_3d(nephrin, wga, collagen4)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        print("Loss: ", loss.clone().detach().to('cpu').numpy())
        #input("Next epoch...")
