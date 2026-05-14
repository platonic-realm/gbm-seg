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
                 _epochs: int,
                 _valid_label_stride: int = 1):

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

        # >1 means the dataset has labels replicated every N slices along Z
        # (e.g. label stacking after Z-interpolation). Validation metrics
        # are then computed only on the "real" slices — those whose
        # (z_start + offset) % N == 0. Loss is unchanged (the model still
        # trains against the full stacked label).
        self.valid_label_stride = int(_valid_label_stride)

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

        # DistributedSampler reshuffles deterministically based on the
        # epoch number — without this, every epoch sees the same shard
        # order, defeating shuffling under DDP.
        sampler = getattr(self.training_loader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(_epoch)
        valid_sampler = getattr(self.validation_loader, 'sampler', None)
        if valid_sampler is not None and hasattr(valid_sampler, 'set_epoch'):
            valid_sampler.set_epoch(_epoch)

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
                                       'train',
                                       train_running_metrics.calculate())

                # Save the snapshot
                self.snapper.save(self.model,
                                  _epoch,
                                  self.stepper.getSteps())

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
        z_start = _data.get('z_start')
        if z_start is not None:
            z_start = z_start.to(self.device)

        with torch.no_grad():

            logits, results = self.model(sample)

            self.visualizer.draw(_channels=sample,
                                 _labels=labels,
                                 _predictions=results,
                                 _epoch_id=_epoch_id,
                                 _batch_id=_batch_id)

            # Loss against the full patch — the model is supervised at every
            # Z slice (including stacked copies), so the training signal must
            # see them too. Only the metric counting changes.
            loss = self.loss_function(logits, labels)

            metric_pred, metric_target = self._select_real_z(
                results, labels, z_start)
            metrics = Metrics(self.no_of_classes,
                              metric_pred,
                              metric_target)

            return metrics.reportMetrics(self.metric_list, loss)

    def _select_real_z(self, _predictions, _labels, _z_start):
        """Restrict prediction/label tensors to real-label Z slices.

        When ``valid_label_stride <= 1`` or ``_z_start`` is missing, returns
        the inputs unchanged. Otherwise builds a (B, Z) boolean mask where
        ``(z_start[b] + offset) % stride == 0`` and flattens prediction +
        label voxels to only the real slices, in (n_real_slices, H, W)
        layout — ``Metrics`` flattens further internally, so shape after
        masking only needs to be consistent between the two tensors.
        """
        if self.valid_label_stride <= 1 or _z_start is None:
            return _predictions, _labels

        # Predictions are argmax outputs (B, Z, H, W); labels match.
        B, Z = _labels.shape[0], _labels.shape[-3]
        z_idx = torch.arange(Z, device=_labels.device)
        # (B, Z) — True where the original-grid Z position is a real label.
        masks = ((_z_start.view(B, 1) + z_idx.view(1, Z))
                 % self.valid_label_stride == 0)

        if not masks.any():
            # Degenerate case (shouldn't happen for stride <= Z): fall back
            # to the unmasked tensors so we still report *something*.
            return _predictions, _labels

        H, W = _labels.shape[-2], _labels.shape[-1]
        flat_mask = masks.flatten()
        pred_flat = _predictions.reshape(B * Z, H, W)
        label_flat = _labels.reshape(B * Z, H, W)
        return pred_flat[flat_mask], label_flat[flat_mask]
