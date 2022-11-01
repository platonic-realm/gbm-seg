"""
Author: Arash Fatehi
Date:   30.10.2022
File:   trainer.py
"""

# Python Imports
import sys
import logging

# Library Imports
import torch
from torch import nn
from tqdm import tqdm

# Local Imports
from src.configs import LOG_LEVEL


def train_3d_unet(_model, _dataloader, _device, _optimizer, _loss_function):

    logging.debug("######################")
    logging.debug("Using device: %s", _device)

    if _device == 'cuda':
        logging.debug("Using data prallel paradigm")
        _model = nn.DataParallel(_model)

    logging.debug("Moving model to device: %s", _device)
    _model.to(_device)

    process_bar = tqdm(_dataloader,
                       file=sys.stdout,
                       disable=LOG_LEVEL == logging.DEBUG)

    for index, data in enumerate(process_bar):
        logging.debug("**********************")
        logging.debug("Batch no: %d", index)

        nephrin = data['nephrin'].to(_device)
        wga = data['wga'].to(_device)
        collagen4 = data['collagen4'].to(_device)
        labels = data['labels'].to(_device)

        if index == 0:
            logging.debug("input channel shape: %s", nephrin.shape)

        logging.debug("Clearing out gradients in the optimizer.")
        _optimizer.zero_grad()

        logging.debug("Inference...")
        outputs = _model(nephrin, wga, collagen4)

        logging.debug("Calculating the loss.")
        loss = _loss_function(outputs, labels)

        logging.debug("Backpropagation...")
        loss.backward()
        _optimizer.step()

        loss_value = loss.item()
        logging.debug("Loss: %.5f", loss_value)

        corrects = (outputs == labels).float().sum().item()

        logging.debug("Corrects: %d", corrects)
        logging.debug("Accuracy: %.4f", corrects/torch.numel(outputs))

        process_bar.set_description(f"Batch-{index}")
        process_bar.set_postfix({'Loss:': loss_value})
