"""
Author: Arash Fatehi
Date:   13.12.2022
"""

# Python Imports
import logging

# Library Imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

# Local Imports
from src.train.trainer import Trainer
from src.models.unet3d_me import Unet3DME
from src.utils.metrics.classification import Metrics


class Unet3DMETrainer(Trainer):
    def __init__(self,
                 _configs: dict,
                 _label_correction_function):

        super().__init__(_configs,
                         _label_correction_function)

        assert self.configs['model']['name'] == 'unet_3d_me', \
               "This class should only be used with unet_3d_me configuration."\
               + f"{self.configs['model']['name']} was given instead."

        self.feature_maps: list = self.configs['model']['feature_maps']
        self.channels: list = self.configs['model']['channels']
        self.metrics: list = self.configs['metrics']

        self.model = Unet3DME(1,
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

        self.step += 1

        if self.ddp:
            device = self.device_id
        else:
            device = self.device

        nephrin = _data['nephrin'].to(device)
        wga = _data['wga'].to(device)
        collagen4 = _data['collagen4'].to(device)
        labels = _data['labels'].to(device)
        labels = labels.long()

        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, results = self.model(nephrin, wga, collagen4)
                loss = self.loss(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, results = self.model(nephrin, wga, collagen4)
            loss = self.loss(logits, labels)
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

        with torch.no_grad():

            logits, results = self.model(nephrin, wga, collagen4)

            sample = self._stack_channels(nephrin, wga, collagen4)
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

            self.gpu_metrics.add(results)

            if self.pytorch_profiling:
                self.prof.step()

            if self.step % freq == 0:
                # We should calculate once and report twice
                metrics = self.gpu_metrics.calculate()
                self._log_tensorboard_metrics(self.step,
                                              'train',
                                              metrics)

                logging.info("Epoch: %d/%d, Batch: %d/%d, Step: %d\n"
                             "Info: %s",
                             _epoch+1,
                             self.epochs,
                             index+1,
                             len(self.training_loader),
                             self.step,
                             metrics)

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
