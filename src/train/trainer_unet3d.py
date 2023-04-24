"""
Author: Arash Fatehi
Date:   23.11.2022
"""

# Python Imports
import logging

# Library Imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

# Local Imports
from src.train.trainer import Trainer
from src.models.unet3d import Unet3D
from src.utils.metrics import Metrics


class Unet3DTrainer(Trainer):
    def __init__(self,
                 _configs: dict,
                 _label_correction_function):

        super().__init__(_configs,
                         _label_correction_function)
        assert self.configs['model']['name'] == 'unet_3d', \
               "This class should only be used with unet_3d configuration." + \
               f"{self.configs['model']['name']} was given instead."

        self.feature_maps: list = self.configs['model']['feature_maps']
        self.channels: list = self.configs['model']['channels']
        self.metrics: list = self.configs['metrics']

        self.model = Unet3D(len(self.channels),
                            self.number_class,
                            _feature_maps=self.feature_maps)

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
        labels = _data['labels'].to(device)
        labels = labels.long()

        sample = self._stack_channels(nephrin, wga, collagen4)

        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, results = self.model(sample)
                loss = self.loss(logits, labels)
        else:
            logits, results = self.model(sample)
            loss = self.loss(logits, labels)

        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        metrics = Metrics(self.number_class,
                          results,
                          labels)

        return self._reports_metrics(metrics, loss)

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

        sample = self._stack_channels(nephrin, wga, collagen4)

        with torch.no_grad():

            logits, results = self.model(sample)

            self._visualize_validation(_epoch_id=_epoch_id,
                                       _batch_id=_batch_id,
                                       _inputs=sample,
                                       _labels=labels,
                                       _predictions=results)

            metrics = Metrics(self.number_class,
                              results,
                              labels)

            loss = self.loss(logits, labels)

            return self._reports_metrics(metrics, loss)

    def _train_epoch(self, _epoch: int):

        freq = self.configs['report_freq']

        if self.pytorch_profiling:
            self.prof.start()

        for index, data in enumerate(self.training_loader):

            results = self._training_step(_epoch,
                                          index,
                                          data)

            self.batch_metrics.add(results)

            if self.pytorch_profiling:
                self.prof.step()

            if index % freq == 0:
                tb_step = _epoch * self.training_batch_size + index + freq
                metrics = self.batch_metrics.calculate()
                logging.info("Epoch: %d/%d, Batch: %d/%d,\n"
                             "Info: %s",
                             _epoch,
                             self.epochs-1,
                             index,
                             len(self.training_loader),
                             metrics)

                self._log_tensorboard_metrics(tb_step,
                                              'train',
                                              metrics)

        if self.pytorch_profiling:
            self.prof.stop()

        self.batch_metrics.calculate()
        for index, data in enumerate(self.validation_loader):

            results = self._validate_step(_epoch_id=_epoch,
                                          _batch_id=index,
                                          _data=data)

            self.batch_metrics.add(results)

            if index % freq == 0:
                tb_step = _epoch * self.validation_batch_size + index + freq
                metrics = self.batch_metrics.calculate()
                logging.info("Validation, Batch: %d/%d,\n"
                             "Info: %s",
                             index,
                             len(self.validation_loader),
                             metrics)

                self._log_tensorboard_metrics(tb_step,
                                              'valid',
                                              metrics)
