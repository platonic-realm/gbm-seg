"""
Author: Arash Fatehi
Date:   05.11.2022
File:   train_unet3d.py
"""
# Python Imprts
import os

# Library Imports
import torch
from torch.utils.data import DataLoader

# Local Imports
from src.models.unet3d import Unet3D
from src.models.unet3d_me import Unet3DME
from src.models.losses import DiceLoss
from src.utils.datasets import GBMDataset
from src.train.trainer import Trainer
from src.configs import VISUAL_OUTPUT_PATH
from src.utils.visual import visualize_predictions


def train_undet3d(_args):

    epochs =_args.epochs
    no_of_channles = len(_args.channels)
    feature_maps = _args.feature_maps
    batch_size = _args.batch_size
    sample_dimension = _args.sample_dimension
    training_ds_path = _args.tds
    validation_ds_path = _args.vds
    validation_no_of_batches = _args.validation_batch_no
    validation_visulization = _args.validation_visualization
    pixel_per_step = _args.pixel_stride
    learning_rate = _args.learning_rate
    data_parallelism = _args.data_parallelism

    training_dataset = GBMDataset(
        _source_directory=training_ds_path,
        _sample_dimension=sample_dimension,
        _pixel_per_step=pixel_per_step
    )

    validation_dataset = GBMDataset(
        _source_directory=validation_ds_path,
        _sample_dimension=sample_dimension,
        _pixel_per_step=pixel_per_step
    )

    training_loader = DataLoader(training_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unet_3d = Unet3D(no_of_channles, _feature_maps=feature_maps)
    optimizer = torch.optim.Adam(unet_3d.parameters(), lr=learning_rate)

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

        sample = torch.cat((nephrin, wga, collagen4),
                           dim=1)

        _optimizer.zero_grad()

        outputs = _model(sample)

        loss = _loss_function(outputs, labels)

        loss.backward()
        _optimizer.step()

        loss_value = loss.item()

        corrects = (outputs == labels).float().sum().item()
        accuracy = corrects/torch.numel(outputs)

        return {'loss': f"{loss_value:.4f}"}, \
               {'corrects': corrects, 'accuracy': f"{accuracy:.4f}"}

    def validation_function(_model,
                            _data,
                            _device,
                            _loss_function,
                            _epoch=None,
                            _batch_id=None):

        nephrin = _data['nephrin'].to(_device)
        wga = _data['wga'].to(_device)
        collagen4 = _data['collagen4'].to(_device)
        labels = _data['labels'].to(_device)

        sample = torch.cat((nephrin, wga, collagen4),
                           dim=1)

        with torch.no_grad():

            outputs = _model(sample)

            if validation_visulization:
                output_dir = os.path.join(VISUAL_OUTPUT_PATH, f"{_epoch:02d}/")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_dir = os.path.join(output_dir, f"{_batch_id:02d}/")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                visualize_predictions(_inputs=sample[0],
                                      _labels=labels[0],
                                      _predictions=outputs[0],
                                      _output_dir=output_dir)

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
                      _validation_no_of_batches=validation_no_of_batches,
                      _optimizer=optimizer,
                      _loss_function=loss_function,
                      _device=device,
                      _data_parallelism=data_parallelism)

    trainer.train(_epochs=epochs)



def train_undet3d_me(_epochs,
                     _no_of_channles,
                     _feature_maps,
                     _batch_size,
                     _sample_dimension,
                     _training_ds_path,
                     _validation_ds_path,
                     _validation_no_of_batches,
                     _pixel_per_step,
                     _learning_rate):

    training_dataset = GBMDataset(
        _source_directory=_training_ds_path,
        _sample_dimension=_sample_dimension,
        _pixel_per_step=_pixel_per_step
    )

    validation_dataset = GBMDataset(
        _source_directory=_validation_ds_path,
        _sample_dimension=_sample_dimension,
        _pixel_per_step=_pixel_per_step
    )

    training_loader = DataLoader(training_dataset,
                                 batch_size=_batch_size,
                                 shuffle=False)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=_batch_size,
                                   shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unet_3d = Unet3DME(_no_of_channles, _feature_maps=_feature_maps)
    optimizer = torch.optim.Adam(unet_3d.parameters(), lr=_learning_rate)
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
                      _validation_no_of_batches=_validation_no_of_batches,
                      _optimizer=optimizer,
                      _loss_function=loss_function,
                      _device=device,
                      _data_parallelism=True)

    trainer.train(_epochs=_epochs)
