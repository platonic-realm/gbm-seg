# Python Imports
from __future__ import annotations

# Library Imports
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.train.snapper import Snapper
from src.train.stepper import StepperInterface

# Local Imports
from src.utils.metrics.clfication import Metrics
from src.utils.metrics.log.metric_logger import MetricLogger
from src.utils.metrics.memory import GPURunningMetrics
from src.utils.visual.training import TrainVisualizer


class Unet3DTrainer:
    """Standard 3D U-Net trainer. Validation runs every ``report_freq`` steps
    (not every epoch); ``ReduceLROnPlateau`` steps on validation Dice."""

    def __init__(self,
                 _model: nn.Module,
                 _loss_funtion,
                 _stepper: StepperInterface,
                 _snapper: Snapper,
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

        self.loss_function = _loss_funtion
        self.stepper = _stepper
        self.snapper = _snapper
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

        _model.to(self.device)
        self.model = _model

        # Track best validation metrics (by Dice) across the run so the
        # CV orchestrator can collect a single best per fold without
        # re-running validation.
        self.best_valid_metrics: dict | None = None
        self.best_valid_epoch: int | None = None
        self.best_valid_step: int | None = None

    def best_metrics(self) -> dict | None:
        """Return a JSON-serialisable snapshot of the best validation
        metrics (and which epoch/step achieved them), or None if no
        validation cycle ran during training."""
        if self.best_valid_metrics is None:
            return None
        return {
            **{k: float(v) for k, v in self.best_valid_metrics.items()},
            'best_epoch': self.best_valid_epoch,
            'best_step': self.best_valid_step,
        }

    def train(self) -> None:
        """Run training for ``self.epochs`` epochs."""
        for epoch in range(0, self.epochs + 1):
            self.trainEpoch(epoch)

    def trainEpoch(self, _epoch: int):

        train_running_metrics = GPURunningMetrics(self.device,
                                                  self.metric_list)

        for index, data in enumerate(self.training_loader):

            # A1: training_loader.dataset is now a plain GBMDataset (no
            # Subset wrapper since we no longer use random_split). The dataset
            # was built with _is_valid=False so the call is redundant but
            # kept for safety against external state mutation.
            self.training_loader.dataset.setIsValid(False)
            results = self.trainStep(_epoch, index, data)
            train_running_metrics.add(results)

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

                self.validation_loader.dataset.setIsValid(True)
                for index, data in enumerate(self.validation_loader):

                    results = self.validStep(_epoch_id=_epoch,
                                             _batch_id=index,
                                             _data=data)

                    valid_running_metrics.add(results)

                metrics = valid_running_metrics.calculate()

                # C1.1: the scheduler is now config-selectable.
                # ReduceLROnPlateau is metric-driven; PolynomialLR (and any
                # future epoch-driven scheduler) takes no argument.
                if _epoch > 0:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(metrics['Dice'])
                    else:
                        self.scheduler.step()

                self.metric_logger.log(_epoch,
                                       self.stepper.getSteps(),
                                       self.stepper.getSeenLabels(),
                                       'valid',
                                       metrics)

                # Update best-valid tracking. Dice is the canonical CV
                # metric (also drives ReduceLROnPlateau); break ties by
                # later epoch since training generally improves.
                current_dice = float(metrics.get('Dice', float('-inf')))
                prior_best = (float('-inf')
                              if self.best_valid_metrics is None
                              else float(self.best_valid_metrics.get(
                                  'Dice', float('-inf'))))
                if current_dice > prior_best:
                    self.best_valid_metrics = dict(metrics)
                    self.best_valid_epoch = _epoch
                    self.best_valid_step = int(self.stepper.getSteps())

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
