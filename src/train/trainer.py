# Python Imports

# Library Imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local Imports
from src.utils.metrics.clfication import Metrics
from src.train.stepper import StepperInterface
from src.train.snapper import Snapper
from src.train.profiler import Profiler
from src.utils.visual.training import TrainVisualizer
from src.utils.metrics.memory import GPURunningMetrics
from src.utils.metrics.log.metric_logger import MetricLogger


class Unet3DTrainer():
    def __init__(self,
                 _model: nn.Module,
                 _loss_funtion,
                 _stepper: StepperInterface,
                 _snapper: Snapper,
                 _profiler: Profiler,
                 _visualizer: TrainVisualizer,
                 _metric_logger: MetricLogger,
                 _lr_scheduler: ReduceLROnPlateau,
                 _training_loader: DataLoader,
                 _validation_loader: DataLoader,
                 _no_of_classes: int,
                 _metric_list: list,
                 _device: str,
                 _freq: int,
                 _epochs: int):

        self.model = _model
        self.loss_function = _loss_funtion
        self.stepper = _stepper
        self.snapper = _snapper
        self.profiler = _profiler
        self.visualizer = _visualizer
        self.metric_logger = _metric_logger
        self.scheduler = _lr_scheduler

        self.training_loader = _training_loader
        self.validation_loader = _validation_loader

        self.no_of_classes = _no_of_classes

        self.metric_list = _metric_list
        self.device = _device
        self.freq = _freq
        self.epochs = _epochs

    def train(self):
        self.model.to(self.device)
        for epoch in range(0, self.epochs + 1):
            self.trainEpoch(epoch)

    def trainEpoch(self, _epoch: int):

        self.profiler.start()

        train_running_metrics = GPURunningMetrics(self.device,
                                                  self.metric_list)

        for index, data in enumerate(self.training_loader):

            results = self.trainStep(_epoch, index, data)
            train_running_metrics.add(results)

            self.profiler.step()

            if self.stepper.getSteps() % self.freq == 0:
                self.metric_logger.log(_epoch,
                                       self.stepper.getSteps(),
                                       self.stepper.getSeenLabels(),
                                       'train',
                                       train_running_metrics.calculate())

                # Save the snapshot
                self.snapper.save(self.model,
                                  _epoch,
                                  self.stepper.getSteps(),
                                  self.stepper.getSeenLabels())

                valid_running_metrics = GPURunningMetrics(self.device,
                                                          self.metric_list)
                for index, data in enumerate(self.validation_loader):

                    results = self.validStep(_epoch_id=_epoch,
                                             _batch_id=index,
                                             _data=data)

                    valid_running_metrics.add(results)

                metrics = valid_running_metrics.calculate()

                # The scheduler changes the lr based on the provided metric
                if _epoch > 0:
                    self.scheduler.step(metrics['Dice'])

                self.metric_logger.log(_epoch,
                                       self.stepper.getSteps(),
                                       self.stepper.getSeenLabels(),
                                       'valid',
                                       metrics)

        self.profiler.stop()

    def trainStep(self,
                  _epoch_id: int,
                  _batch_id: int,
                  _data: dict) -> dict:

        sample = _data['sample'].to(self.device)
        labels = _data['labels'].to(self.device).long()

        logits, results, loss = self.stepper.step(sample, labels)

        metrics = Metrics(self.no_of_classes,
                          results,
                          labels)

        return metrics.reportMetrics(self.metric_list, loss)

    def validStep(self,
                  _epoch_id: int,
                  _batch_id: int,
                  _data: dict) -> dict:

        sample = _data['sample'].to(self.device)
        labels = _data['labels'].to(self.device).long()

        with torch.no_grad():

            logits, results = self.model(sample)

            self.visualizer.draw(_channels=sample,
                                 _labels=labels,
                                 _predictions=results,
                                 _epoch_id=_epoch_id,
                                 _batch_id=_batch_id)

            metrics = Metrics(self.no_of_classes,
                              results,
                              labels)

            loss = self.loss_function(logits, labels)

            return metrics.reportMetrics(self.metric_list, loss)
