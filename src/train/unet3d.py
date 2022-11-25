"""
Author: Arash Fatehi
Date:   23.11.2022
"""

# Python Imports
import sys
import logging
import os

# Library Imports
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

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
        self.metrics = self.configs['metrics']

        self.visualization = self.configs['visualization']
        self.visualization_path = self.configs['visualization_path']

        self.validation_limit = self.configs['valid_ds']['batch_limit']
        self.model = Unet3D(len(self.channels),
                            _feature_maps=self.feature_maps)

        if self.ddp:
            init_process_group(backend="nccl")

        self._load_snapshot()

        if self.device == 'cuda':
            self.model.to(self.device_id)
            logging.info("Moving model to gpu %d", self.device_id)
        else:
            self.model.to(self.device)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self._prepare_data()
        self._prepare_optimizer()
        self._prepare_loss()

    def __del__(self):
        if self.ddp:
            destroy_process_group()

    def _save_sanpshot(self, epoch: int) -> None:
        if self.snapshot_path is None:
            return
        if self.local_rank > 0:
            return

        snapshot = {}
        snapshot['EPOCHS'] = epoch
        if self.ddp:
            snapshot['MODEL_STATE'] = self.model.module.state_dict()
        else:
            snapshot['MODEL_STATE'] = self.model.state_dict()

        torch.save(snapshot, self.snapshot_path)
        logging.info("Snapshot saved on epoch %d", epoch)

    def _load_snapshot(self) -> None:
        if not os.path.exists(self.snapshot_path):
            return

        snapshot = torch.load(self.snapshot_path)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epoch_resume = snapshot['EPOCHS']
        logging.info("Resuming training at epoch: %d", self.epoch_resume)

    def _log_tensorboard(self) -> None:
        print('dummy')

    def _prepare_data(self) -> None:

        training_ds_dir: str = self.configs['train_ds']['path']
        training_sample_dimension: list = \
            self.configs['train_ds']['sample_dimension']
        training_pixel_stride: list = \
            self.configs['train_ds']['pixel_stride']
        training_dataset = GBMDataset(
                _source_directory=training_ds_dir,
                _sample_dimension=training_sample_dimension,
                _pixel_per_step=training_pixel_stride)

        validation_ds_dir: str = self.configs['valid_ds']['path']
        validation_sample_dimension: list = \
            self.configs['valid_ds']['sample_dimension']
        validation_pixel_stride: list = \
            self.configs['valid_ds']['pixel_stride']
        validation_dataset = GBMDataset(
                _source_directory=validation_ds_dir,
                _sample_dimension=validation_sample_dimension,
                _pixel_per_step=validation_pixel_stride)

        if self.ddp:
            train_sampler = DistributedSampler(training_dataset)
            valid_sampler = DistributedSampler(validation_dataset)
        else:
            train_sampler = None
            valid_sampler = None

        training_batch_size: int = self.configs['train_ds']['batch_size']
        training_shuffle: bool = self.configs['train_ds']['shuffle']
        self.training_loader = DataLoader(training_dataset,
                                          batch_size=training_batch_size,
                                          shuffle=training_shuffle,
                                          sampler=train_sampler)

        validation_batch_size: int = self.configs['valid_ds']['batch_size']
        validation_shuffle: bool = self.configs['valid_ds']['shuffle']
        self.validation_loader = DataLoader(validation_dataset,
                                            batch_size=validation_batch_size,
                                            shuffle=validation_shuffle,
                                            sampler=valid_sampler)

    def _prepare_optimizer(self) -> None:
        optimizer_name: str = self.configs['optim']['name']
        if optimizer_name == 'adam':
            lr: float = self.configs['optim']['lr']
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=lr)

    def _prepare_loss(self) -> None:
        loss_name: str = self.configs['loss']
        if loss_name == 'DiceLoss':
            self.loss = DiceLoss()

    def _training_step(self, _data: dict) -> (dict, dict):

        if self.ddp:
            device = self.device_id
        else:
            device = self.device

        nephrin = _data['nephrin'].to(device)
        wga = _data['wga'].to(device)
        collagen4 = _data['collagen4'].to(device)
        labels = _data['labels'].to(device)

        sample = torch.cat((nephrin, wga, collagen4),
                           dim=1)

        self.optimizer.zero_grad()

        outputs = self.model(sample)

        loss = self.loss(outputs, labels)
        loss.backward()

        self.optimizer.step()

        loss_value = loss.item()

        corrects = (outputs == labels).float().sum().item()
        accuracy = corrects/torch.numel(outputs)

        return {'loss': f"{loss_value:.4f}"}, \
               {'corrects': corrects, 'accuracy': f"{accuracy:.3f}"}

    def _validate_step(self, _data: dict) -> (dict, dict):

        if self.ddp:
            device = self.device_id
        else:
            device = self.device

        nephrin = _data['nephrin'].to(device)
        wga = _data['wga'].to(device)
        collagen4 = _data['collagen4'].to(device)
        labels = _data['labels'].to(device)

        sample = torch.cat((nephrin, wga, collagen4),
                           dim=1)

        with torch.no_grad():

            outputs = self.model(sample)

            loss = self.loss(outputs, labels)
            loss_value = loss.item()

            corrects = (outputs == labels).float().sum().item()
            accuracy = corrects/torch.numel(outputs)

            return {'loss': f"{loss_value:.4f}"}, \
                   {'corrects': corrects, 'accuracy': f"{accuracy:.3f}"}

    def _train_epoch(self, _epoch: int):

        if self.tqdm:
            train_enum = tqdm(self.training_loader,
                              file=sys.stdout)
            train_enum.set_description(f"Epoch {_epoch+1}/{self.epochs}")
        else:
            train_enum = self.training_loader

        for data in train_enum:
            losses, metrics = self._training_step(data)

            if self.tqdm:
                postfix = {}
                for key in losses:
                    postfix[key] = losses[key]
                for key in metrics:
                    postfix[key] = metrics[key]

                train_enum.set_postfix(postfix)

        if self.tqdm:
            valid_enum = tqdm(self.validation_loader,
                              file=sys.stdout,
                              total=self.validation_limit)
            valid_enum.set_description("Validation")
        else:
            valid_enum = self.validation_loader

        for index, data in enumerate(valid_enum):

            if index >= self.validation_limit:
                break

            losses, metrics = self._validate_step(data)

            if self.tqdm:
                postfix = {}
                for key in losses:
                    postfix[key] = losses[key]
                for key in metrics:
                    postfix[key] = metrics[key]

                valid_enum.set_postfix(postfix)

