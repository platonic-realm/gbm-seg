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
from src.utils.datasets import GBMDataset
from src.trainer import Trainer

if __name__ == '__main__':

    logging.basicConfig(
                    level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("testing.log"),
                              logging.StreamHandler(sys.stdout)])

    training_dataset = GBMDataset(
        _source_directory='/data/afatehi/data/GBM-Train-DS',
        _sample_dimension=(6, 256, 256),
        _pixel_per_step=(1, 16, 16)
    )

    validation_dataset = GBMDataset(
        _source_directory='/data/afatehi/data/GBM-Valid-DS',
        _sample_dimension=(6, 256, 256),
        _pixel_per_step=(1, 16, 16)
    )

    training_loader = DataLoader(training_dataset,
                                 batch_size=20,
                                 shuffle=False)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=20,
                                   shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unet_3d = Unet3DME(1, _feature_maps=(32, 64, 128))
    optimizer = torch.optim.SGD(unet_3d.parameters(), lr=0.0001, momentum=0.9)
    loss_function = DiceLoss()

    def training_funtion(_model,
                         _index,
                         _data,
                         _device,
                         _optimizer,
                         _loss_function):

        nephrin = _data['nephrin'].to(_device)
        wga = _data['wga'].to(_device)
        collagen4 = _data['collagen4'].to(_device)
        labels = _data['labels'].to(_device)

        _optimizer.zero_grad()

        outputs = _model(nephrin, wga, collagen4)

        loss = _loss_function(outputs, labels)

        loss.backward()
        _optimizer.step()

        loss_value = loss.item()

        corrects = (outputs == labels).float().sum().item()
        accuracy = corrects/torch.numel(outputs)

        return {'loss': f"{loss_value:.4f}"}, \
               {'corrects': corrects, 'accuracy': f"{accuracy:.3f}"}

    def validation_function(_model,
                            _data,
                            _device,
                            _loss_function):

        nephrin = _data['nephrin'].to(_device)
        wga = _data['wga'].to(_device)
        collagen4 = _data['collagen4'].to(_device)
        labels = _data['labels'].to(_device)

        with torch.no_grad():

            outputs = _model(nephrin, wga, collagen4)

            loss = _loss_function(outputs, labels)

            loss_value = loss.item()

            corrects = (outputs == labels).float().sum().item()
            accuracy = corrects/torch.numel(outputs)

            return {'loss': f"{loss_value:.4f}"}, \
                   {'corrects': corrects, 'accuracy': f"{accuracy:.3f}"}

    trainer = Trainer(_model=unet_3d,
                      _training_function=training_funtion,
                      _training_dataloader=training_loader,
                      _validation_function=validation_function,
                      _validation_dataloader=validation_loader,
                      _validation_no_of_batches=500,
                      _optimizer=optimizer,
                      _loss_function=loss_function,
                      _device=device,
                      _data_parallelism=True,
                      _number_of_gpus=2)

    trainer.train(_epochs=10)

    # train_3d_unet(_model=unet_3d,
    #               _dataloader=training_loader,
    #               _device=device,
    #               _optimizer=optimizer,
    #               _loss_function=loss_function)
