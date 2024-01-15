# Python Imports
import os
import re
from pathlib import Path

# Library Imports
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Local Imports
from src.train.stepper import StepperSimple, StepperMixedPrecision, StepperInterface
from src.train.snapper import Snapper
from src.train.profiler import Profiler
from src.train.trainer import Unet3DTrainer
from src.train.losses.loss_dice import DiceLoss
from src.train.losses.loss_iou import IoULoss
from src.models.unet3d.unet3d import Unet3D
from src.data.ds_base import BaseDataset
from src.data.ds_train import GBMDataset
from src.utils.visual.training import TrainVisualizer
from src.utils.visual.painter import GIFPainter3D, TIFPainter3D
from src.utils.metrics.log.metric_sql import MetricSQL
from src.utils.metrics.log.metric_tboard import MetricTensorboard
from src.utils.metrics.log.metric_logger import MetricLogger
from src.utils.misc import blind_test
from src.data.ds_infer import InferenceDataset
from src.infer.morph import Morph
from src.infer.inference import Inference


class Factory:

    def __init__(self, _configs: dict):
        self.configs = _configs
        self.root_path = _configs['root_path']
        self.result_path = os.path.join(self.root_path,
                                        self.configs['trainer']['result_path'])

    def createModel(self,
                    _no_of_channles: int,
                    _no_of_classes: int,
                    _result_shape: list,
                    _inference: bool = False) -> nn.Module:

        model = Unet3D(self.configs['trainer']['model']['name'],
                       _no_of_channles,
                       _no_of_classes,
                       _encoder_kernel_size=self.configs['trainer']['model']['encoder_kernel'],
                       _decoder_kernel_size=self.configs['trainer']['model']['decoder_kernel'],
                       _feature_maps=self.configs['trainer']['model']['feature_maps'],
                       _sample_dimension=self.configs['trainer']['train_ds']['sample_dimension'],
                       _inference=_inference,
                       _result_shape=_result_shape)

        return model

    def createLoss(self):
        device = self.configs['trainer']['device']
        weights = torch.tensor(self.configs['trainer']['loss_weights']).to(device)

        loss_name: str = self.configs['trainer']['loss']
        if loss_name == 'Dice':
            loss = DiceLoss(_weights=weights)
        if loss_name == 'IoU':
            loss = IoULoss(_weights=weights)
        if loss_name == 'CrossEntropy':
            loss = nn.CrossEntropyLoss(weight=weights)

        return loss

    def createOptimizer(self, _model: nn.Module):
        optimizer_name: str = self.configs['trainer']['optim']['name']
        if optimizer_name == 'adam':
            lr: float = self.configs['trainer']['optim']['lr']
            optimizer = torch.optim.Adam(_model.parameters(),
                                         lr=lr)
        return optimizer

    def createScheduler(self, _optimizer):
        return ReduceLROnPlateau(_optimizer, mode='max', verbose=True)

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
                      _profiler: Profiler,
                      _visualizer: TrainVisualizer,
                      _metric_logger: MetricLogger,
                      _lr_scheduler: ReduceLROnPlateau,
                      _training_loader: DataLoader,
                      _validation_loader: DataLoader,
                      _no_of_classes: int):

        metrics_list = self.configs['trainer']['metrics']
        device = self.configs['trainer']['device']
        report_freq = self.configs['trainer']['report_freq']
        epochs = self.configs['trainer']['epochs']
        dp = self.configs['trainer']['dp']

        if self.configs['trainer']['model']['name'] == 'unet_3d':
            trainer = Unet3DTrainer(_model,
                                    _loss_function,
                                    _stepper,
                                    _snapper,
                                    _profiler,
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
                                    dp)
        else:
            assert False, "Please provide a valid model name in the config file"

        return trainer

    def createTrainDataset(self) -> BaseDataset:

        root_path = self.configs['root_path']

        training_ds_dir: str = os.path.join(root_path,
                                            self.configs['trainer']['train_ds']['path'])
        training_sample_dimension: list = self.configs['trainer']['train_ds']['sample_dimension']
        training_pixel_stride: list = self.configs['trainer']['train_ds']['pixel_stride']

        training_augmentation = None
        if self.configs['trainer']['train_ds']['augmentation']['enabled']:
            training_augmentation = self.configs['trainer']['train_ds']['augmentation']['methods']

        training_dataset = GBMDataset(
                _source_directory=training_ds_dir,
                _sample_dimension=training_sample_dimension,
                _pixel_per_step=training_pixel_stride,
                _ignore_stride_mismatch=self.configs['trainer']['train_ds']['ignore_stride_mismatch'],
                _augmentation=training_augmentation,
                _augmentation_workers=self.configs['trainer']['train_ds']['augmentation']['workers'])

        return training_dataset

    def createTrainDataLoader(self, _training_dataset: BaseDataset) -> DataLoader:

        training_batch_size: int = self.configs['trainer']['train_ds']['batch_size']
        training_shuffle: bool = self.configs['trainer']['train_ds']['shuffle']
        training_pin_memory: bool = self.configs['trainer']['train_ds']['pin_memory']

        training_loader = DataLoader(_training_dataset,
                                     batch_size=training_batch_size,
                                     shuffle=training_shuffle,
                                     pin_memory=training_pin_memory,
                                     num_workers=self.configs['trainer']['train_ds']['workers'])

        return training_loader

    def createValidDataset(self) -> BaseDataset:

        root_path = self.configs['root_path']

        validation_ds_dir: str = os.path.join(root_path,
                                              self.configs['trainer']['valid_ds']['path'])
        validation_sample_dimension: list = self.configs['trainer']['valid_ds']['sample_dimension']
        validation_pixel_stride: list = self.configs['trainer']['valid_ds']['pixel_stride']

        validation_dataset = GBMDataset(
                _source_directory=validation_ds_dir,
                _sample_dimension=validation_sample_dimension,
                _pixel_per_step=validation_pixel_stride,
                _ignore_stride_mismatch=self.configs['trainer']['valid_ds']['ignore_stride_mismatch'])

        return validation_dataset

    def createValidDataLoader(self, _validation_dataset) -> DataLoader:

        validation_batch_size: int = self.configs['trainer']['valid_ds']['batch_size']
        validation_shuffle: bool = self.configs['trainer']['valid_ds']['shuffle']
        validation_pin_memory: bool = self.configs['trainer']['valid_ds']['pin_memory']

        validation_loader = DataLoader(_validation_dataset,
                                       batch_size=validation_batch_size,
                                       shuffle=validation_shuffle,
                                       pin_memory=validation_pin_memory,
                                       num_workers=self.configs['trainer']['valid_ds']['workers'])

        return validation_loader

    def createProfiler(self):

        save_path = os.path.join(self.root_path,
                                 self.result_path,
                                 self.configs['trainer']['profiling']['path'])

        enabled = self.configs['trainer']['profiling']['enabled']

        scheduler_wait = self.configs['trainer']['profiling']['scheduler']['wait']
        scheduler_warmup = self.configs['trainer']['profiling']['scheduler']['warmup']
        scheduler_active = self.configs['trainer']['profiling']['scheduler']['active']
        scheduler_repeat = self.configs['trainer']['profiling']['scheduler']['repeat']

        profile_memory = self.configs['trainer']['profiling']['profile_memory']
        record_shapes = self.configs['trainer']['profiling']['record_shapes']
        with_flops = self.configs['trainer']['profiling']['with_flops']
        with_stack = self.configs['trainer']['profiling']['with_stack']

        save_tensorboard = self.configs['trainer']['profiling']['save']['tensorboard']
        save_text = self.configs['trainer']['profiling']['save']['text']
        save_std = self.configs['trainer']['profiling']['save']['print']

        profiler = Profiler(enabled,
                            save_path,
                            scheduler_wait,
                            scheduler_warmup,
                            scheduler_active,
                            scheduler_repeat,
                            profile_memory,
                            record_shapes,
                            with_flops,
                            with_stack,
                            save_tensorboard,
                            save_text,
                            save_std)

        return profiler

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

    def createMetricLogger(self,
                           _model: nn.Module,
                           _valid_dataloader: DataLoader,
                           _loss_function,
                           _no_of_classes: int):

        device = self.configs['trainer']['device']

        sqlite_enabled = self.configs['trainer']['sqlite']
        database_path = os.path.join(self.root_path,
                                     'report.db')

        metric_sql = MetricSQL(database_path) if sqlite_enabled else None

        tboard_enabled = self.configs['trainer']['tensorboard']['enabled']
        tboard_seen_label: bool = self.configs['trainer']['tensorboard']['label_seen']

        tensorboard_path = Path(os.path.join(self.root_path,
                                self.result_path,
                                self.configs['trainer']['tensorboard']['path']))

        metric_tboard = None

        if tboard_enabled:
            metrics_list = self.configs['trainer']['metrics']
            zero_metrics = blind_test(_model,
                                      _valid_dataloader,
                                      _loss_function,
                                      device,
                                      _no_of_classes,
                                      metrics_list)

            metric_tboard = MetricTensorboard(tensorboard_path,
                                              zero_metrics,
                                              tboard_seen_label)

        metric_logger = MetricLogger(metric_sql, metric_tboard)

        return metric_logger

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

        for file_name in directory_content:
            file_path = os.path.join(source_directory, file_name)
            dataset = InferenceDataset(
                    _file_path=file_path,
                    _sample_dimension=sample_dimension,
                    _pixel_per_step=pixel_stride,
                    _scale_factor=scale_factor,
                    _no_of_classes=no_of_classes)

            data_loader = DataLoader(dataset,
                                     batch_size=self.configs['inference']['inference_ds']['batch_size'],
                                     pin_memory=pin_memory,
                                     shuffle=False)

            result.append(data_loader)

        return result

    def createMorphModule(self):
        morph = Morph(_device=self.configs['trainer']['device'],
                      _ave_kernel_size=5)
        return morph


    def createInferer(self,
                      _model: nn.Module,
                      _data_loaders: list,
                      _morph: Morph,
                      _snapper: Snapper):

        device = self.configs['trainer']['device']
        result_path = self.configs['inference']['result_dir']
        snapshot_path = self.configs['inference']['snapshot_path']
        dp = self.configs['trainer']['dp']

        post_processing = self.configs['inference']['post_processing']['enabled']
        psp_obj_min_size = self.configs['inference']['post_processing']['min_size']
        psp_kernel_size = self.configs['inference']['post_processing']['kernel_size']

        inferer = Inference(_model,
                            _data_loaders,
                            _morph,
                            _snapper,
                            device,
                            result_path,
                            snapshot_path,
                            dp,
                            post_processing,
                            psp_obj_min_size,
                            psp_kernel_size)

        return inferer
