"""
Author: Arash Fatehi
Date:   23.11.2022
"""

# Python Imports

# Library Imports
import torch
from torch.utils.data import DataLoader

# Local Imports
from src.train.trainer import Trainer
from src.datasets.train_ds import GBMDataset
from src.models.unet3d.unet3d import Unet3D
from src.models.unet3d.losses import DiceLoss


class Unet3DTrainer(Trainer):
    def __init__(self, _configs: dict):
        super().__init__(_configs)
        assert self.configs['model']['name'] == 'unet_3d', \
               "This class should only be used with unet_3d configuration." + \
               f"{self.configs['model']['name']} was given instead."

        self.feature_maps = self.configs['model']['feature_maps']
        self.channels = self.configs['model']['channels']
        self.metrics = self.configs['model']['metrics']

        self.visualization = self.configs['model']['visualization']
        self.visualization_path = self.configs['model']['visualization_path']

        self.model = Unet3D(len(self.channels),
                            _feature_maps=self.feature_maps)

        self.__prepare_data()
        self.__load_snapshot()
        self.__prepare_optimizer()
        self.__prepare_loss()

    def __save_sanpshot(self) -> None:
        pass

    def __load_snapshot(self) -> None:
        pass

    def __log_tensorboard(self) -> None:
        pass

    def __prepare_data(self) -> None:

        training_ds_dir: str = self.configs['train_ds']['path']
        training_sample_dimension: list = \
            self.configs['train_ds']['sample_dimension']
        training_pixel_stride: list = \
            self.configs['train_ds']['pixel_stride']
        training_dataset = GBMDataset(
                _source_directory=training_ds_dir,
                _sample_dimension=training_sample_dimension,
                _pixel_per_step=training_pixel_stride
                )

        validation_ds_dir: str = self.configs['valid_ds']['path']
        validation_sample_dimension: list = \
            self.configs['valid_ds']['sample_dimension']
        validation_pixel_stride: list = \
            self.configs['valid_ds']['pixel_stride']
        validation_dataset = GBMDataset(
                _source_directory=validation_ds_dir,
                _sample_dimension=validation_sample_dimension,
                _pixel_per_step=validation_pixel_stride
                )

        training_batch_size: int = self.configs['train_ds']['batch_size']
        training_shuffle: bool = self.configs['train_ds']['shuffle']
        self.training_loader = DataLoader(training_dataset,
                                          batch_size=training_batch_size,
                                          shuffle=training_shuffle)

        validation_batch_size: int = self.configs['valid_ds']['batch_size']
        validation_shuffle: bool = self.configs['valid_ds']['shuffle']
        self.validation_loader = DataLoader(validation_dataset,
                                            batch_size=validation_batch_size,
                                            shuffle=validation_shuffle)

    def __prepare_optimizer(self) -> None:
        optimizer_name: str = self.configs['optim']['name']
        if optimizer_name == 'adam':
            lr: float = self.configs['optim']['lr']
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=lr)

    def __prepare_loss(self) -> None:
        loss_name: str = self.configs['loss']
        if loss_name == 'DiceLoss':
            self.loss = DiceLoss()

    def __training_step(self) -> (dict, dict):

        nephrin = _data['nephrin'].to(self.device)
        wga = _data['wga'].to(self.device)

    def __validate_step(self) -> (dict, dict):
        pass
