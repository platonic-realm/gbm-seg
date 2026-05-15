# Python Imports
from __future__ import annotations

import logging
import os
import random
import re

# Library Imports
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.ds_base import BaseDataset
from src.data.ds_infer import InferenceDataset
from src.data.ds_train import GBMDataset
from src.infer.inference import Inference
from src.infer.morph import Morph
from src.infer.psp import PSP
from src.models import build_model
from src.train.distributed import (
    ddp_requested,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
)
from src.train.losses.loss_compound import CompoundLoss
from src.train.losses.loss_cont import ContLoss
from src.train.losses.loss_dice import DiceLoss
from src.train.losses.loss_ds import DeepSupervisionLoss
from src.train.losses.loss_iou import IoULoss
from src.train.snapper import Snapper

# Local Imports
from src.train.stepper import StepperInterface, StepperMixedPrecision, StepperSimple
from src.train.trainer import Unet3DTrainer
from src.utils.metrics.log.metric_logger import MetricLogger
from src.utils.visual.painter import GIFPainter3D, TIFPainter3D
from src.utils.visual.training import TrainVisualizer


def _worker_init_fn(worker_id: int):
    # Each worker forks with the parent's RNG state; reseed numpy/random with
    # a stable per-worker offset so workers don't draw identical samples.
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


class Factory:
    """Central wiring layer — every component (model, dataset, loss, optimizer,
    stepper, snapper, visualizer, profiler, metric logger, inferer, PSP, morph)
    is built here from the experiment's config dict.
    """

    def __init__(self, _configs: dict):
        self.configs = _configs
        self.root_path = _configs['root_path']
        self.result_path = os.path.join(self.root_path,
                                        self.configs['trainer']['result_path'])

    def createModel(self,
                    _no_of_channles: int,
                    _no_of_classes: int) -> nn.Module:
        """Build the segmentation model from ``configs.trainer.model``.

        B1.2 refactor: dispatch through :mod:`src.models` registry so the
        factory is decoupled from any single model class. Each registered
        model's ``build`` callable handles its own config-to-constructor
        translation. See ``src/models/__init__.py``.
        """
        model = build_model(
            self.configs['trainer']['model']['name'],
            self.configs,
            _no_of_channles,
            _no_of_classes)

        # DDP path takes precedence over DP when both flags are set.
        # Required launcher env: `torchrun --nproc-per-node=N gbm.py train …`
        # (see sbatch/train_ddp.sbatch). DDP supports torch.compile,
        # channels_last_3d, and cudnn.benchmark — all gated by config.
        use_ddp = ddp_requested(self.configs) and is_distributed()
        if use_ddp:
            local_rank = get_local_rank()
            device = torch.device(f'cuda:{local_rank}')
            model = model.to(device)

            if self.configs['trainer'].get('channels_last_3d', False):
                try:
                    model = model.to(memory_format=torch.channels_last_3d)
                    logging.info("rank=%d: channels_last_3d enabled",
                                 get_rank())
                except Exception as exc:
                    logging.warning(
                        "rank=%d: channels_last_3d failed (%s); "
                        "continuing with default memory layout.",
                        get_rank(), exc)

            # compile-then-DDP-wrap is the documented working order
            # (unlike DP, where the wrap-then-compile dance is required).
            if self.configs['trainer'].get('compile', False):
                try:
                    model = torch.compile(model)
                    logging.info("rank=%d: torch.compile applied (DDP path)",
                                 get_rank())
                except Exception as exc:
                    logging.warning(
                        "rank=%d: torch.compile failed (%s); eager mode.",
                        get_rank(), exc)

            model = DDP(model, device_ids=[local_rank],
                        find_unused_parameters=False)
            return model

        # DP path (unchanged — the 3D U-Net runs continue to use this).
        # JIT-compile (Inductor) when the config enables it. ORDER MATTERS:
        # `DP(torch.compile(model))` raises `AttributeError: '<ModelClass>'
        # object has no attribute 'encoder_layers'` on the first forward
        # because DataParallel's replica scattering bypasses the OptimizedModule
        # attribute proxy. The supported workaround is to DP-wrap first and
        # then compile the inner `.module`. DDP doesn't have this problem.
        if self.configs['trainer']['dp']:
            model = DP(model)

        if self.configs['trainer'].get('compile', False):
            try:
                if isinstance(model, DP):
                    model.module = torch.compile(model.module)
                else:
                    model = torch.compile(model)
                logging.info("torch.compile() applied to model "
                             "'%s' (mode=default).",
                             self.configs['trainer']['model']['name'])
            except Exception as exc:
                logging.warning(
                    "torch.compile() failed (%s); continuing in eager mode.",
                    exc)

        return model

    def _build_single_loss(self, name: str, weights, params: dict = None):
        """Construct one of the registered single-loss classes by name.

        Sub-losses inside ``CompoundLoss`` come through here too, so adding
        a new single-loss class means a one-line addition both here and in
        the recognised name list — no factory branching beyond.
        """
        params = params or {}
        if name == 'Dice':
            return DiceLoss(_weights=weights)
        if name == 'IoU':
            return IoULoss(_weights=weights)
        if name == 'CrossEntropy':
            return nn.CrossEntropyLoss(weight=weights)
        if name == 'Cont':
            cont_alpha = params.get('cont_alpha',
                                    self.configs['trainer'].get('cont_alpha', 0.7))
            cont_beta = params.get('cont_beta',
                                   self.configs['trainer'].get('cont_beta', 0.3))
            return ContLoss(nn.CrossEntropyLoss(weight=weights),
                            _alpha=cont_alpha, _beta=cont_beta)
        raise NotImplementedError(f"Unknown loss: {name!r}")

    def createLoss(self):
        device = self.configs['trainer']['device']
        weights = torch.tensor(self.configs['trainer']['loss_weights']).to(device)

        loss_name: str = self.configs['trainer']['loss']
        if loss_name == 'Compound':
            components = []
            for entry in self.configs['trainer']['compound_loss']:
                sub_loss = self._build_single_loss(
                    entry['name'], weights, entry.get('params', {}))
                components.append((sub_loss, float(entry.get('weight', 1.0))))
            loss = CompoundLoss(components)
        else:
            loss = self._build_single_loss(loss_name, weights)

        loss = loss.to(device) if isinstance(loss, nn.Module) else loss

        # C1.2: wrap with deep supervision if the model is producing
        # multi-resolution logits. The wrapper degrades transparently when
        # the model returns a single tensor, so wrapping is safe regardless.
        ds_cfg = self.configs['trainer'].get('deep_supervision', {})
        if ds_cfg.get('enabled', False):
            ds_weights = ds_cfg.get('weights')  # optional explicit list
            loss = DeepSupervisionLoss(loss, weights=ds_weights)

        return loss

    def createOptimizer(self, _model: nn.Module, _loss):
        """Build the optimiser from ``configs.trainer.optim``.

        C1.1: supports ``adam`` (existing) and ``sgd`` (nnU-Net default —
        momentum=0.99, often with poly-decay scheduler). Each branch reads
        its own keyword arguments from the same ``optim`` sub-dict; unknown
        keys are ignored.
        """
        optim_cfg = self.configs['trainer']['optim']
        name = optim_cfg['name']
        lr = optim_cfg['lr']

        if isinstance(_loss, nn.Module):
            parameters = list(_model.parameters()) + list(_loss.parameters())
        else:
            parameters = _model.parameters()

        if name == 'adam':
            return torch.optim.Adam(parameters, lr=lr)
        if name == 'sgd':
            momentum = optim_cfg.get('momentum', 0.99)
            weight_decay = optim_cfg.get('weight_decay', 0.0)
            nesterov = optim_cfg.get('nesterov', False)
            return torch.optim.SGD(parameters, lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay,
                                   nesterov=nesterov)
        raise NotImplementedError(
            f"Unknown optimiser: {name!r}. Supported: 'adam', 'sgd'.")

    def createScheduler(self, _optimizer):
        """Build the LR scheduler from ``configs.trainer.scheduler``.

        Supports ``reduce_on_plateau`` (default, existing), ``poly_decay``
        (nnU-Net's ``(1 - t/T)^power``), and the disabled form — set
        ``trainer.scheduler: null`` or ``trainer.scheduler: {name: none}``
        to skip LR scheduling entirely (the trainer guards None internally).

        The trainer dispatches its ``.step()`` call accordingly — see
        :class:`Unet3DTrainer.trainEpoch`.

        For backwards compatibility, if ``trainer.scheduler`` is absent the
        scheduler defaults to ``reduce_on_plateau`` with the prior
        (mode=max) behaviour.
        """
        sched_cfg = self.configs['trainer'].get('scheduler', {})
        # `trainer.scheduler: null` in yaml deserialises to None; treat
        # that as the disabled form (returns None — no scheduler).
        if sched_cfg is None:
            return None
        name = sched_cfg.get('name', 'reduce_on_plateau')

        if name == 'none':
            return None
        if name == 'reduce_on_plateau':
            mode = sched_cfg.get('mode', 'max')
            factor = sched_cfg.get('factor', 0.1)
            patience = sched_cfg.get('patience', 10)
            return ReduceLROnPlateau(_optimizer, mode=mode,
                                     factor=factor, patience=patience)
        if name == 'poly_decay':
            # nnU-Net: lr_t = lr_0 * (1 - t/T) ** power. `total_iters`
            # is in scheduler-step units (here: per validation cycle).
            total_iters = sched_cfg.get('total_iters',
                                        self.configs['trainer']['epochs'])
            power = sched_cfg.get('power', 0.9)
            return torch.optim.lr_scheduler.PolynomialLR(
                _optimizer, total_iters=total_iters, power=power)
        raise NotImplementedError(
            f"Unknown scheduler: {name!r}. Supported: "
            "'reduce_on_plateau', 'poly_decay'.")

    def createStepper(self,
                      _model: nn.Module,
                      _optimizer,
                      _loss):

        if self.configs['trainer']['mixed_precision']:
            return StepperMixedPrecision(_model, _optimizer, _loss)
        else:
            return StepperSimple(_model, _optimizer, _loss)

    def createSnapper(self):
        snapshot_path = os.path.join(self.root_path,
                                     self.result_path,
                                     self.configs['trainer']['snapshot_path'])

        snapper = Snapper(snapshot_path)
        return snapper

    def createTrainer(self,
                      _model: nn.Module,
                      _loss_function,
                      _stepper: StepperInterface,
                      _snapper: Snapper,
                      _visualizer: TrainVisualizer,
                      _metric_logger: MetricLogger,
                      _lr_scheduler: ReduceLROnPlateau,
                      _training_loader: DataLoader,
                      _validation_loader: DataLoader,
                      _no_of_classes: int,
                      _starting_epoch: int = 0,
                      _starting_best_metrics: dict | None = None,
                      _starting_best_epoch: int | None = None,
                      _starting_best_step: int | None = None):

        metrics_list = self.configs['trainer']['metrics']
        device = self.configs['trainer']['device']
        report_freq = self.configs['trainer']['report_freq']
        epochs = self.configs['trainer']['epochs']

        # `gbm.py create` z-upsamples every channel and stacks the label
        # along Z by ``trainer.z_scale`` (written into the per-experiment
        # configs.yaml). Validation metrics are masked to slices where
        # Z % z_scale == 0 — i.e. the original-label positions — so the
        # same label voxel isn't counted z_scale times. Default 1 = no
        # interpolation, no masking (every slice is a real label).
        valid_label_stride = int(
            self.configs['trainer'].get('z_scale', 1) or 1)

        # Snapshot restore happens in train.main_train *before* this call,
        # so all state objects (model, optimiser, scheduler, stepper) are
        # already warm here. Resume-related fields (starting_epoch + best
        # tracker) are threaded through from main_train.

        # B1.2 refactor: the trainer is model-agnostic — any model that
        # satisfies the post-A3 interface (forward(x) -> (logits, outputs))
        # can be trained by Unet3DTrainer. The name reflects the original
        # use case; the class is effectively generic now.
        trainer = Unet3DTrainer(_model,
                                _loss_function,
                                _stepper,
                                _snapper,
                                _visualizer,
                                _metric_logger,
                                _lr_scheduler,
                                _training_loader,
                                _validation_loader,
                                _no_of_classes,
                                metrics_list,
                                device,
                                report_freq,
                                epochs,
                                valid_label_stride,
                                _starting_epoch,
                                _starting_best_metrics,
                                _starting_best_epoch,
                                _starting_best_step)

        return trainer

    def createTrainDataset(self, file_filter=None) -> BaseDataset:
        """Build the *training* dataset (augmentation on).

        A1: ``file_filter`` (a list of TIFF basenames) restricts the
        dataset to the per-fold training subset. When None, every TIFF
        in ``train_ds.path`` is used (legacy behaviour, used only by
        old experiments that lack fold_assignments.yaml).
        """
        root_path = self.configs['root_path']
        training_ds_dir: str = os.path.join(root_path,
                                            self.configs['trainer']['train_ds']['path'])
        training_sample_dimension: list = self.configs['trainer']['train_ds']['sample_dimension']
        training_pixel_stride: list = self.configs['trainer']['train_ds']['pixel_stride']

        training_augmentation_offline = None
        if self.configs['trainer']['train_ds']['augmentation']['enabled_offline']:
            training_augmentation_offline = self.configs['trainer']['train_ds']['augmentation']['methods_offline']

        training_augmentation_online = None
        if self.configs['trainer']['train_ds']['augmentation']['enabled_online']:
            training_augmentation_online = self.configs['trainer']['train_ds']['augmentation']['methods_online']

        training_dataset = GBMDataset(
                _source_directory=training_ds_dir,
                _sample_dimension=training_sample_dimension,
                _pixel_per_step=training_pixel_stride,
                _ignore_stride_mismatch=self.configs['trainer']['train_ds']['ignore_stride_mismatch'],
                _augmentation_offline=training_augmentation_offline,
                _augmentation_online=training_augmentation_online,
                _augmentation_workers=self.configs['trainer']['train_ds']['augmentation']['workers'],
                _is_valid=False,
                _file_filter=file_filter)

        return training_dataset

    def createValidDataset(self, file_filter=None) -> BaseDataset:
        """Build the *validation* dataset (augmentation off, is_valid=True).

        A1: shares the ds_train directory with the training dataset, but
        is restricted to the per-fold validation subset via ``file_filter``
        and runs without augmentation. Patches are sampled the same way as
        training (same sample_dimension + pixel_stride).
        """
        root_path = self.configs['root_path']
        training_ds_dir: str = os.path.join(root_path,
                                            self.configs['trainer']['train_ds']['path'])
        training_sample_dimension: list = self.configs['trainer']['train_ds']['sample_dimension']
        # Validation uses a coarser stride (matches the pre-A1 config field
        # `valid_ds.pixel_stride`) — denser stride is for training only.
        validation_pixel_stride: list = self.configs['trainer']['valid_ds']['pixel_stride']

        validation_dataset = GBMDataset(
                _source_directory=training_ds_dir,
                _sample_dimension=training_sample_dimension,
                _pixel_per_step=validation_pixel_stride,
                _ignore_stride_mismatch=self.configs['trainer']['valid_ds']['ignore_stride_mismatch'],
                _augmentation_offline=None,
                _augmentation_online=None,
                _augmentation_workers=0,
                _is_valid=True,
                _file_filter=file_filter)

        return validation_dataset

    def createTrainDataLoader(self, _training_dataset: BaseDataset) -> DataLoader:

        training_batch_size: int = self.configs['trainer']['train_ds']['batch_size']
        training_shuffle: bool = self.configs['trainer']['train_ds']['shuffle']
        training_pin_memory: bool = self.configs['trainer']['train_ds']['pin_memory']
        num_workers = self.configs['trainer']['train_ds']['workers']

        # Under DDP each rank sees a disjoint shard of the dataset; the
        # DistributedSampler enforces the partition and handles per-epoch
        # shuffling. Loader's own `shuffle=` must be False when a sampler
        # is supplied. Single-process (or DP) path keeps the original
        # behaviour.
        sampler = None
        if is_distributed():
            sampler = DistributedSampler(
                _training_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=training_shuffle,
                drop_last=False)
            training_shuffle = False

        training_loader = DataLoader(
            _training_dataset,
            batch_size=training_batch_size,
            shuffle=training_shuffle,
            sampler=sampler,
            pin_memory=training_pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            worker_init_fn=_worker_init_fn)

        return training_loader

    def createValidDataLoader(self, _validation_dataset) -> DataLoader:

        validation_batch_size: int = self.configs['trainer']['train_ds']['batch_size']
        validation_shuffle: bool = False  # never shuffle the validation set
        validation_pin_memory: bool = self.configs['trainer']['train_ds']['pin_memory']
        num_workers = self.configs['trainer']['train_ds']['workers']

        # Under DDP each rank validates a disjoint shard. Set shuffle=False
        # on the sampler so the partition is stable across epochs.
        sampler = None
        if is_distributed():
            sampler = DistributedSampler(
                _validation_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
                drop_last=False)

        validation_loader = DataLoader(
            _validation_dataset,
            batch_size=validation_batch_size,
            shuffle=validation_shuffle,
            sampler=sampler,
            pin_memory=validation_pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            worker_init_fn=_worker_init_fn)

        return validation_loader

    def createVisualizer(self):

        enabled = self.configs['trainer']['visualization']['enabled']
        chance = self.configs['trainer']['visualization']['chance']
        save_as_tif = self.configs['trainer']['visualization']['tif']
        save_as_gif = self.configs['trainer']['visualization']['gif']

        v_path = os.path.join(self.root_path,
                              self.result_path,
                              self.configs['trainer']['visualization']['path'])

        gif_painter = GIFPainter3D() if save_as_gif else None
        tif_painter = TIFPainter3D() if save_as_tif else None

        visualizer = TrainVisualizer(enabled,
                                     chance,
                                     gif_painter,
                                     tif_painter,
                                     v_path)

        return visualizer

    def createMetricLogger(self):
        # E2: optional W&B backend. The wandb run itself is initialised in
        # train.py:maybe_init_wandb so the run config + name are visible from
        # the start; this just creates the per-step log dispatcher.
        wandb_cfg = self.configs['trainer'].get('wandb', {})
        metric_wandb = None
        if wandb_cfg.get('enabled', False):
            try:
                from src.utils.metrics.log.metric_wandb import MetricWandb
                metric_wandb = MetricWandb()
            except ImportError:
                logging.warning(
                    "trainer.wandb.enabled=True but wandb is not installed; "
                    "skipping W&B logging. `pip install wandb` to enable.")

        return MetricLogger(metric_wandb)

    def createInferenceDataLoaders(self) -> list:

        result = []

        source_directory = os.path.join(self.root_path,
                                        self.configs['inference']['inference_ds']['path'])
        directory_content = os.listdir(source_directory)
        directory_content = list(filter(lambda _x: re.match(r'(.+).(tiff|tif)', _x),
                                        directory_content))

        no_of_classes = self.configs['inference']['number_class']
        sample_dimension = self.configs['inference']['inference_ds']['sample_dimension']
        pixel_stride = self.configs['inference']['inference_ds']['pixel_stride']
        pin_memory = self.configs['inference']['inference_ds']['pin_memory']
        scale_factor = self.configs['inference']['inference_ds']['scale_factor']
        interpolate = self.configs['inference']['interpolate']

        for file_name in directory_content:
            file_path = os.path.join(source_directory, file_name)
            dataset = InferenceDataset(
                    _file_path=file_path,
                    _sample_dimension=sample_dimension,
                    _pixel_per_step=pixel_stride,
                    _scale_factor=scale_factor,
                    _interpolate=interpolate,
                    _no_of_classes=no_of_classes)

            data_loader = DataLoader(dataset,
                                     batch_size=self.configs['inference']['inference_ds']['batch_size'],
                                     pin_memory=pin_memory,
                                     shuffle=False)

            result.append(data_loader)

        return result

    def createMorphModule(self):
        kernel_size = self.configs['inference']['morph']['kernel_ave_size']
        morph = Morph(_device=self.configs['trainer']['device'],
                      _ave_kernel_size=kernel_size)
        return morph

    def createPSPer(self):

        kernel_size = self.configs['inference']['post_processing']['kernel_size']
        min_2d_size = self.configs['inference']['post_processing']['min_2d_size']
        min_3d_size = self.configs['inference']['post_processing']['min_3d_size']

        psp = PSP(kernel_size,
                  min_2d_size,
                  min_3d_size)

        return psp

    def createInferer(self,
                      _model: nn.Module,
                      _data_loader: DataLoader,
                      _morph: Morph,
                      _snapper: Snapper):
        """Build an :class:`Inference` for one DataLoader.

        A3 refactor: the inferer now owns the sliding-window accumulator
        (``StitchAccumulator``); the stitching mode is read from
        ``configs.inference.stitching``. ``patch_size`` and ``result_shape``
        come from the data_loader's dataset.
        """
        device = self.configs['trainer']['device']
        result_path = self.configs['inference']['result_dir']
        snapshot_path = self.configs['inference']['snapshot_path']
        dp = self.configs['trainer']['dp']

        interpolate = self.configs['inference']['interpolate']
        scale_factor = self.configs['inference']['inference_ds']['scale_factor']
        stitching_mode = self.configs['inference'].get('stitching', 'gaussian')

        # The dataset knows both the per-patch sample dimension (Z, X, Y)
        # and the full-volume result shape (C, Z, X, Y) it expects from
        # the stitched output.
        patch_size = self.configs['inference']['inference_ds']['sample_dimension']
        result_shape = _data_loader.dataset.getResultShape()

        inferer = Inference(_model,
                            _data_loader,
                            _snapper,
                            device,
                            result_path,
                            snapshot_path,
                            dp,
                            interpolate,
                            scale_factor,
                            stitching_mode,
                            patch_size,
                            result_shape)

        return inferer
