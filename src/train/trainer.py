# Python Imports
from __future__ import annotations

# Library Imports
import torch
import torch.distributed as dist
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
                 _valid_label_stride: int = 1,
                 _starting_epoch: int = 0,
                 _starting_best_metrics: dict | None = None,
                 _starting_best_epoch: int | None = None,
                 _starting_best_step: int | None = None):

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

        # Resume hooks — set by main_train from the snapshot's resume info
        # so a continued run starts at the right epoch and doesn't reset
        # the best-validation tracker. Defaults match the from-scratch path.
        self.starting_epoch = int(_starting_epoch)
        self.best_valid_metrics: dict | None = _starting_best_metrics
        self.best_valid_epoch: int | None = _starting_best_epoch
        self.best_valid_step: int | None = _starting_best_step

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
        """Run training from ``self.starting_epoch`` to ``self.epochs``.

        On a fresh run ``starting_epoch == 0`` and the loop covers
        ``0..epochs`` inclusive (the existing N+1 sweep). On a resumed
        run ``starting_epoch`` is set to ``saved_epoch + 1`` by
        ``main_train`` so the already-completed epochs are skipped.
        """
        for epoch in range(self.starting_epoch, self.epochs + 1):
            self.trainEpoch(epoch)

    def trainEpoch(self, _epoch: int):

        # DistributedSampler reshuffles deterministically based on the
        # epoch number — without this, every epoch sees the same shard
        # order, defeating shuffling under DDP.
        sampler = getattr(self.training_loader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(_epoch)
        if self.validation_loader is not None:
            valid_sampler = getattr(self.validation_loader, 'sampler', None)
            if valid_sampler is not None and hasattr(valid_sampler,
                                                     'set_epoch'):
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

                # Snapshot the full resume-able state, not just the model.
                # Pull the optimiser off the stepper (the trainer doesn't
                # hold a direct reference). best_valid_* may still be None
                # this early — the snapshot stores None and main_train's
                # resume path handles None gracefully.
                self.snapper.save(
                    self.model,
                    _epoch,
                    self.stepper.getSteps(),
                    _stepper=self.stepper,
                    _optimizer=getattr(self.stepper, 'optimizer', None),
                    _scheduler=self.scheduler,
                    _best_valid_metrics=self.best_valid_metrics,
                    _best_valid_epoch=self.best_valid_epoch,
                    _best_valid_step=self.best_valid_step,
                )

                # All-data training mode has no validation set — skip the
                # validation cycle, its metric-driven scheduler step, and
                # the best-valid tracker entirely.
                if self.validation_loader is not None:
                    self._validate(_epoch)

        # Epoch-driven schedulers (poly_decay and anything that isn't
        # ReduceLROnPlateau) advance once per epoch at the end of every
        # trainEpoch — same behaviour in all-data mode and in k-fold CV.
        # The total_iters of poly_decay is set in epoch units in both
        # modes. ReduceLROnPlateau is metric-driven and steps inside
        # _validate when validation runs.
        if (self.scheduler is not None
                and not isinstance(self.scheduler, ReduceLROnPlateau)
                and _epoch > 0):
            self.scheduler.step()

    def _validate(self, _epoch: int) -> None:
        """Run one validation cycle: validate over the validation loader,
        step a metric-driven scheduler, log valid metrics, and update the
        best-valid tracker. Only called when a validation loader exists."""
        valid_running_metrics = GPURunningMetrics(self.device,
                                                  self.metric_list)

        self.validation_loader.dataset.setIsValid(True)
        for index, data in enumerate(self.validation_loader):

            results = self.validStep(_epoch_id=_epoch,
                                     _batch_id=index,
                                     _data=data)

            valid_running_metrics.add(results)

        metrics = valid_running_metrics.calculate()

        # Only ReduceLROnPlateau steps inside _validate — it's metric-driven
        # and needs the validation Dice to decide whether to decay. Other
        # schedulers (poly_decay, etc.) are epoch-driven and step once at
        # the end of trainEpoch in both modes, so both branches advance the
        # LR identically across the run.
        if (_epoch > 0
                and self.scheduler is not None
                and isinstance(self.scheduler, ReduceLROnPlateau)):
            self.scheduler.step(metrics['Dice'])

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
        z_start = _data.get('z_start')
        if z_start is not None:
            z_start = z_start.to(self.device)

        logits, results, loss = self.stepper.step(sample, labels)

        # Visualize the augmented training patches actually fed to the
        # model. In all-data mode there is no validation cycle, so
        # trainStep is the only place this data can be captured. Rank 0
        # only — all DDP ranks share the epoch/batch index space and would
        # otherwise race on the same visuals/epoch-N/batch-M/ files.
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.visualizer.draw(_channels=sample,
                                 _labels=labels,
                                 _predictions=results,
                                 _epoch_id=_epoch_id,
                                 _batch_id=_batch_id)

        metric_pred, metric_target = self._select_real_z(
            results, labels, z_start)
        metrics = Metrics(self.no_of_classes,
                          metric_pred,
                          metric_target)

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
            # see them too. Only the metric counting changes (same convention
            # as trainStep — both train and valid metrics are computed on
            # real-label slices only via _select_real_z, while the loss term
            # uses every slice).
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
