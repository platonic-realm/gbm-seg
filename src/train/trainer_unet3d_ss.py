"""
Author: Arash Fatehi
Date:   19.02.2022
"""

# Python Imports
import os
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
from src.utils.metrics.classification import Metrics
from src.utils.metrics.memory import GPURunningMetrics
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
        self.unlabeled_metrics = GPURunningMetrics(self.configs,
                                                   self.device,
                                                   ["loss"])
        self._prepare_unlabeled_data()

    def _training_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> dict:
        self.step += 1

        if self.ddp:
            device = self.device_id
        else:
            device = self.device

        nephrin = _data['nephrin'].to(device)
        wga = _data['wga'].to(device)
        collagen4 = _data['collagen4'].to(device)

        labels = None
        supervised = False
        self.loss.unsupervised()
        if 'labels' in _data:
            labels = _data['labels'].to(device)
            labels = labels.long()
            self.loss.supervised()
            supervised = True

        frames = self._stack_channels(nephrin, wga, collagen4)

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
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, results, interpolation = self.model(sample)
            loss = self.loss(_epoch_id,
                             logits,
                             interpolation,
                             labels,
                             frames)
            loss.backward()
            self.optimizer.step()

        if supervised:
            metrics = Metrics(self.number_class,
                              results,
                              labels)

            return self._reports_metrics(metrics, loss)

        return {'loss': loss}

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
        labels = labels.long()

        frames = self._stack_channels(nephrin, wga, collagen4)

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

            metrics = Metrics(self.number_class,
                              results,
                              labels)

            return self._reports_metrics(metrics, loss)

    def _train_epoch(self, _epoch: int):

        freq = self.configs['report_freq']

        labeled_length = len(self.training_loader)
        unlabeled_length = len(self.unlabeled_loader)

        labeled_iterator = iter(self.training_loader)
        unlabeled_iterator = iter(self.unlabeled_loader)

        ratio = unlabeled_length // labeled_length

        assert ratio >= 2, "It doesn't make sense to have a low ratio."

        train_index = 0
        batch_size = ratio * labeled_length

        if freq > ratio:
            freq = freq + (freq % ratio)
        else:
            freq = ratio

        if self.pytorch_profiling:
            self.prof.start()

        while train_index < batch_size:
            supervised = (train_index % ratio) == 0
            if supervised:
                data = next(labeled_iterator)
            else:
                data = next(unlabeled_iterator)

            train_index += 1

            results = self._training_step(_epoch,
                                          train_index,
                                          data)
            if supervised:
                self.gpu_metrics.add(results)
            else:
                self.unlabeled_metrics.add(results)

            if self.pytorch_profiling:
                self.prof.step()

            if train_index % freq == 0:
                # We should calculate once and report twice
                metrics = self.gpu_metrics.calculate()
                self._log_tensorboard_metrics(self.step,
                                              'train',
                                              metrics)

                logging.info("Epoch: %d/%d, Batch: %d/%d, Step: %d\n"
                             "Info: %s",
                             _epoch+1,
                             self.epochs,
                             train_index,
                             batch_size,
                             self.step,
                             metrics)

                # Save the snapshot
                self._save_sanpshot(_epoch)

                for index, data in enumerate(self.validation_loader):
                    results = self._validate_step(_epoch_id=_epoch,
                                                  _batch_id=index,
                                                  _data=data)
                    self.gpu_metrics.add(results)

                # We should calculate once and report twice
                metrics = self.gpu_metrics.calculate()
                logging.info("Validation, Step: %d\n"
                             "Info: %s",
                             self.step,
                             metrics)

                self._log_tensorboard_metrics(self.step,
                                              'valid',
                                              metrics)

        if self.pytorch_profiling:
            self.prof.stop()

    def _prepare_unlabeled_data(self) -> None:

        unlabeled_ds_dir: str = \
            os.path.join(self.root_path,
                         self.configs['unlabeled_ds']['path'])
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
                                           sampler=unlabeled_sampler,
                                           num_workers=self.configs
                                           ['unlabeled_ds']['workers'])
