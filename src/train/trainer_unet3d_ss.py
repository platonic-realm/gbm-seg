"""
Author: Arash Fatehi
Date:   19.02.2022
"""

# Python Imports
import logging

# Library Imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Local Imports
from src.train.trainer import Trainer
from src.models.unet3d_ss import Unet3DSS
from src.utils.misc import RunningMetric
from src.utils.metrics import Metrics
from src.data.ds_train import GBMDataset
from src.data.ds_train import DatasetType


class Unet3DSemiTrainer(Trainer):
    def __init__(self,
                 _configs: dict,
                 _label_correction_function):

        super().__init__(_configs,
                         _label_correction_function)
        assert self.configs['model']['name'] == 'unet_3d_ss', \
            "This class should only be used with unet_3d_ss configuration." + \
            f"{self.configs['model']['name']} was given instead."

        self.feature_maps: list = self.configs['model']['feature_maps']
        self.channels: list = self.configs['model']['channels']
        self.metrics: list = self.configs['metrics']
        self.sample_dimension = self.configs['train_ds']['sample_dimension']

        self.model = Unet3DSS(len(self.channels),
                              self.number_class,
                              _feature_maps=self.feature_maps,
                              _sample_dimension=self.sample_dimension)

        self._load_snapshot()

        if self.device == 'cuda':
            self.model.to(self.device_id)
            logging.info("Moving model to gpu %d", self.device_id)
        else:
            self.model.to(self.device)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        if self.dp:
            self.model = DP(self.model)

        self._prepare_optimizer()
        self._prepare_loss()

        self.unlabeled_loader = None
        self._prepare_unlabeled_data()

    def _training_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> dict:
        if self.ddp:
            device = self.device_id
        else:
            device = self.device

        if self.skip_training:
            return {'loss': 0,
                    'corrects': 0,
                    'accuracy': 0}

        nephrin = _data['nephrin'].to(device)
        wga = _data['wga'].to(device)
        collagen4 = _data['collagen4'].to(device)

        labels = None
        supervised = False
        self.loss.unsupervised()
        if 'labels' in _data:
            labels = _data['labels'].to(device)
            self.loss.supervised()
            supervised = True

        frames = self._stack_channels(nephrin, wga, collagen4)

        nephrin = nephrin[:, :, ::2, :, :]
        wga = wga[:, :, ::2, :, :]
        collagen4 = collagen4[:, :, ::2, :, :]

        sample = self._stack_channels(nephrin, wga, collagen4)

        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, results, interpolation = self.model(sample)
                loss = self.loss(_epoch_id,
                                 logits,
                                 interpolation,
                                 labels,
                                 frames)
        else:
            logits, results, interpolation = self.model(sample)
            loss = self.loss(_epoch_id,
                             logits,
                             interpolation,
                             labels,
                             frames)

        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        loss_value = loss.item()

        if supervised:
            metrics = Metrics(self.number_class,
                              results,
                              labels)

            corrects = (results == labels).float().sum().item()
            accuracy = metrics.Accuracy()
        else:
            corrects = 0.0
            accuracy = 0.0

        return {'loss': loss_value,
                'corrects': corrects,
                'accuracy': accuracy}

    def _validate_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> dict:

        if self.ddp:
            device = self.device_id
        else:
            device = self.device

        nephrin = _data['nephrin'].to(device)
        wga = _data['wga'].to(device)
        collagen4 = _data['collagen4'].to(device)
        labels = _data['labels'].to(device)

        frames = self._stack_channels(nephrin, wga, collagen4)

        nephrin = nephrin[:, :, ::2, :, :]
        wga = wga[:, :, ::2, :, :]
        collagen4 = collagen4[:, :, ::2, :, :]

        sample = self._stack_channels(nephrin, wga, collagen4)

        self.loss.supervised()

        with torch.no_grad():

            logits, results, interpolation = self.model(sample)

            self._visualize_validation(_epoch_id=_epoch_id,
                                       _batch_id=_batch_id,
                                       _inputs=sample,
                                       _labels=labels,
                                       _predictions=results)

            loss = self.loss(_epoch_id,
                             logits,
                             interpolation,
                             labels,
                             frames)

            loss_value = loss.item()

            corrects = (results == labels).float().sum().item()
            accuracy = corrects/torch.numel(results)

            return {'loss': loss_value,
                    'corrects': corrects,
                    'accuracy': accuracy}

    def _train_epoch(self, _epoch: int):

        train_accuracy = RunningMetric()
        train_loss = RunningMetric()
        valid_accuracy = RunningMetric()
        valid_loss = RunningMetric()

        freq = self.configs['report_freq']

        labeled_length = len(self.training_loader)
        unlabeled_length = len(self.unlabeled_loader)

        labeled_iterator = iter(self.training_loader)
        unlabeled_iterator = iter(self.unlabeled_loader)

        ratio = unlabeled_length // labeled_length

        index = 0
        batch_size = ratio*labeled_length
        while index < batch_size:

            batch_accuracy = RunningMetric()
            batch_loss = RunningMetric()

            if (index % ratio) == 0:
                data = next(labeled_iterator)
            else:
                data = next(unlabeled_iterator)

            index += 1

            results = self._training_step(_epoch,
                                          index,
                                          data)

            # These are used to calculate per epoch metrics
            train_accuracy.add(results['accuracy'])
            train_loss.add(results['loss'])
            # These ones are for in batch calculation
            batch_accuracy.add(results['accuracy'])
            batch_loss.add(results['loss'])

            if index % freq == 0:
                logging.info("Epoch: %d/%d, Batch: %d/%d, "
                             "Loss: %.3f, Accuracy: %.3f",
                             _epoch,
                             self.epochs-1,
                             index,
                             batch_size,
                             batch_loss.calcualte(),
                             batch_accuracy.calcualte())

        for index, data in enumerate(self.validation_loader):

            batch_accuracy = RunningMetric()
            batch_loss = RunningMetric()

            results = self._validate_step(_epoch_id=_epoch,
                                          _batch_id=index,
                                          _data=data)

            # These are used to calculate per epoch metrics
            valid_accuracy.add(results['accuracy'])
            valid_loss.add(results['loss'])
            # These ones are for in batch calculation
            batch_accuracy.add(results['accuracy'])
            batch_loss.add(results['loss'])

            if index % freq == 0:
                logging.info("Validation, Batch: %d/%d, "
                             "Loss: %.3f, Accuracy: %.3f",
                             index,
                             len(self.validation_loader),
                             batch_loss.calcualte(),
                             batch_accuracy.calcualte())

        self._log_tensorboard_metrics(
               _train_accuracy=train_accuracy.calcualte(),
               _train_loss=train_loss.calcualte(),
               _valid_accuracy=valid_accuracy.calcualte(),
               _valid_loss=valid_loss.calcualte(),
               _n_iter=_epoch)

    def _prepare_unlabeled_data(self) -> None:

        unlabeled_ds_dir: str = self.configs['unlabeled_ds']['path']
        unlabeled_sample_dimension: list = \
            self.configs['unlabeled_ds']['sample_dimension']
        unlabeled_pixel_stride: list = \
            self.configs['unlabeled_ds']['pixel_stride']
        unlabeled_channel_map: list = \
            self.configs['unlabeled_ds']['channel_map']
        unlabeled_dataset = GBMDataset(
                _source_directory=unlabeled_ds_dir,
                _sample_dimension=unlabeled_sample_dimension,
                _pixel_per_step=unlabeled_pixel_stride,
                _channel_map=unlabeled_channel_map,
                _dataset_type=DatasetType.Unsupervised,
                _ignore_stride_mismatch=self.configs['unlabeled_ds']
                ['ignore_stride_mismatch'])

        if self.ddp:
            unlabeled_sampler = DistributedSampler(unlabeled_dataset)
        else:
            unlabeled_sampler = None

        unlabeled_batch_size: int = self.configs['unlabeled_ds']['batch_size']
        unlabeled_shuffle: bool = self.configs['unlabeled_ds']['shuffle']
        self.unlabeled_loader = DataLoader(unlabeled_dataset,
                                           batch_size=unlabeled_batch_size,
                                           shuffle=unlabeled_shuffle,
                                           sampler=unlabeled_sampler)
