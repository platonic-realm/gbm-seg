"""
Author: Arash Fatehi
Date:   11.10.2023
"""

# Python Imports
import logging

# Library Imports
import torch
from torch.nn.parallel import DataParallel as DP

# Local Imports
from src.train.trainer import Trainer
from src.models.swin_unetr import SwinUNETR
from src.utils.metrics.classification import Metrics


class SwinUNETRTrainer(Trainer):
    def __init__(self,
                 _configs: dict,
                 _label_correction_function=None):
        super().__init__(_configs,
                         _label_correction_function)
        assert self.configs['model']['name'] == 'swinunetr', \
               "This class should only be used with swinunetr configuration." + \
               f"{self.configs['model']['name']} was given instead."

        self.feature_maps: list = self.configs['model']['feature_maps']
        self.channels: list = self.configs['model']['channels']
        self.metrics: list = self.configs['metrics']

        self.model = SwinUNETR(img_size=self.configs['train_ds']['sample_dimension'],
                               in_channels=len(self.channels),
                               out_channels=self.number_class)

        self._load_snapshot()

        if self.device == 'cuda':
            self.model.to(self.device_id)
            logging.info("Moving model to gpu %d", self.device_id)
        else:
            self.model.to(self.device)

        if self.dp:
            self.model = DP(self.model)

        self._prepare_optimizer()
        self._prepare_loss()

    def _training_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> dict:

        self.step += 1

        device = self.device

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
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, results = self.model(sample)
            loss = self.loss(logits, labels)
            loss.backward()
            self.optimizer.step()

        self.seen_labels += self.training_batch_size
        metrics = Metrics(self.number_class,
                          results,
                          labels)

        return self._reports_metrics(metrics, loss)

    def _validate_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> dict:

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
                                       _batch_id=self.step,
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
                self._log_metrics(_epoch,
                                  self.step,
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

                # Save the snapshot
                self._save_sanpshot(_epoch)

                for index, data in enumerate(self.validation_loader):

                    results = self._validate_step(_epoch_id=_epoch,
                                                  _batch_id=index,
                                                  _data=data)

                    self.gpu_metrics.add(results)

                # We should calculate once and report twice
                metrics = self.gpu_metrics.calculate()

                # The scheduler changes the lr based on the provided metric
                if _epoch > 0:
                    self.scheduler.step(metrics['Dice'])

                logging.info("Validation, Step: %d\n"
                             "Info: %s",
                             self.step,
                             metrics)

                self._log_metrics(_epoch,
                                  self.step,
                                  'valid',
                                  metrics)
                self._log_tensorboard_metrics(self.step,
                                              'valid',
                                              metrics)
                self._log_tensorboard_metrics(self.seen_labels,
                                              'valid_ls',
                                              metrics)

        if self.pytorch_profiling:
            self.prof.stop()
