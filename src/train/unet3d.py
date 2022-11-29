"""
Author: Arash Fatehi
Date:   23.11.2022
"""

# Python Imports
import sys
import logging
import os
import random
from pathlib import Path

# Library Imports
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local Imports
from src.train.trainer import Trainer
from src.datasets.train_ds import GBMDataset
from src.models.unet3d.unet3d import Unet3D
from src.models.unet3d.losses import DiceLoss
from src.utils.visual import visualize_predictions
from src.utils.misc import RunningAverage


class Unet3DTrainer(Trainer):
    def __init__(self, _configs: dict):
        super().__init__(_configs)
        assert self.configs['model']['name'] == 'unet_3d', \
               "This class should only be used with unet_3d configuration." + \
               f"{self.configs['model']['name']} was given instead."

        self.feature_maps: list = self.configs['model']['feature_maps']
        self.channels: list = self.configs['model']['channels']
        self.metrics: list = self.configs['metrics']

        self.validation_limit: int = self.configs['valid_ds']['batch_limit']
        self.model = Unet3D(len(self.channels),
                            _feature_maps=self.feature_maps)

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

    def _log_tensorboard_metrics(self,
                                 _n_iter: int,
                                 _train_accuracy: float,
                                 _train_loss: float,
                                 _valid_accuracy: float,
                                 _valid_loss: float) -> None:
        if not self.tensorboard:
            return
        if not self.configs['tensorboard']['metrics']:
            return
        if self.ddp and self.rank != 0:
            return


        tb_writer = SummaryWriter(self.tensorboard_path.resolve())

        tb_writer.add_scalar('Accuracy/train', _train_accuracy, _n_iter)
        tb_writer.add_scalar('Loss/train', _train_loss, _n_iter)

        tb_writer.add_scalar('Accuracy/valid', _valid_accuracy, _n_iter)
        tb_writer.add_scalar('Loss/valid', _valid_loss, _n_iter)

        tb_writer.close()

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

    def _validate_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> (dict, dict):

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

            self._visualize_validation(_epoch_id=_epoch_id,
                                       _batch_id=_batch_id,
                                       _inputs=sample,
                                       _labels=labels,
                                       _predictions=outputs)

            loss = self.loss(outputs, labels)
            loss_value = loss.item()

            corrects = (outputs == labels).float().sum().item()
            accuracy = corrects/torch.numel(outputs)

            return {'loss': f"{loss_value:.4f}"}, \
                   {'corrects': corrects, 'accuracy': f"{accuracy:.3f}"}

    def _train_epoch(self, _epoch: int):

        train_accuracy = RunningAverage()
        train_loss = RunningAverage()
        valid_accuracy = RunningAverage()
        valid_loss = RunningAverage()

        if self.tqdm:
            train_enum = tqdm(self.training_loader,
                              file=sys.stdout)
            train_enum.set_description(f"Epoch {_epoch+1}/{self.epochs}")
        else:
            train_enum = self.training_loader

        for index, data in enumerate(train_enum):
            losses, metrics = self._training_step(data)

            # Calculating an writing average of metrics
            train_accuracy.add(metrics['accuracy'])
            train_loss.add(losses['loss'])

            if self.tqdm:
                postfix = {}
                for key in losses:
                    postfix[key] = losses[key]
                for key in metrics:
                    postfix[key] = metrics[key]

                train_enum.set_postfix(postfix)

        if self.tqdm:
            validation_limit = min(len(self.validation_loader),
                                   self.validation_limit)
            valid_enum = tqdm(self.validation_loader,
                              file=sys.stdout,
                              total=validation_limit)
            valid_enum.set_description("Validation")
        else:
            valid_enum = self.validation_loader

        for index, data in enumerate(valid_enum):

            if index >= self.validation_limit:
                break

            losses, metrics = self._validate_step(_epoch_id=_epoch,
                                                  _batch_id=index,
                                                  _data=data)

            # Calculating an writing average of metrics
            valid_accuracy.add(metrics['accuracy'])
            valid_loss.add(losses['loss'])

            if self.tqdm:
                postfix = {}
                for key in losses:
                    postfix[key] = losses[key]
                for key in metrics:
                    postfix[key] = metrics[key]

                valid_enum.set_postfix(postfix)

        self._log_tensorboard_metrics(
               _train_accuracy=train_accuracy.calcualte(),
               _train_loss=train_loss.calcualte(),
               _valid_accuracy=valid_accuracy.calcualte(),
               _valid_loss=valid_loss.calcualte(),
               _n_iter=_epoch)

    def _visualize_validation(self,
                              _epoch_id: int,
                              _batch_id: int,
                              _inputs,
                              _labels,
                              _predictions):

        dice = random.random()
        if dice > self.visualization_chance:
            return

        batch_size = len(_inputs)
        sample_id = random.randint(0, batch_size-1)

        if self.ddp:
            path: Path = Path(self.configs['visualization']['path'] +
                              f"/worker-{self.rank:02}/{_epoch_id}" +
                              f"/{_batch_id}/{sample_id}/")
        else:
            path: Path = Path(self.configs['visualization']['path'] +
                              f"/{_epoch_id}/{_batch_id}/{sample_id}/")

        path.mkdir(parents=True, exist_ok=True)

        enable_gif: bool = self.configs['visualization']['gif']
        enable_tif: bool = self.configs['visualization']['tif']
        enable_mesh: bool = self.configs['visualization']['mesh']

        output_dir: str = path.resolve()
        visualize_predictions(_inputs=_inputs[sample_id],
                              _labels=_labels[sample_id],
                              _predictions=_predictions[sample_id],
                              _output_dir=output_dir,
                              _produce_gif_files=enable_gif,
                              _produce_tif_files=enable_tif,
                              _produce_3d_model=enable_mesh)
