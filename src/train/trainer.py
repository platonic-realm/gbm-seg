"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imports
# Python's wierd implementation of abstract methods
from abc import ABC, abstractmethod
from pathlib import Path
import os
import glob
import random
import logging
from datetime import datetime

# Libary Imports
import torch
from torch import Tensor, nn
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import ProfilerActivity

# Local Imports
from src.utils.misc import create_dirs_recursively, create_config_tag
from src.utils.visual import VisualizerUnet3D
from src.data.ds_train import GBMDataset
from src.utils.losses.loss_dice import DiceLoss
from src.utils.losses.loss_ss import SelfSupervisedLoss


# Tip for using abstract methods in python... dont use
# double __ for the abstract method as python name
# mangeling will mess them and you are going to have a hard time
class Trainer(ABC):
    def __init__(self,
                 _configs: dict,
                 _label_correction_function=None):
        self.configs: dict = _configs['trainer']

        # these variables are declared because some methods need
        # them, but they will be defined in the subclasses
        self.model = None
        self.optimizer = None
        self.loss = None
        self.no_of_classes = None

        self.label_correction = _label_correction_function

        self.model_tag = create_config_tag(_configs)

        self.root_path = _configs['root_path']

        # Note we are using self.configs from now on ...
        self.model_name = self.configs['model']['name']
        self.epochs: int = self.configs['epochs']
        self.epoch_resume = 0
        self.save_interval = self.configs['save_interval']
        self.result_path = os.path.join(self.root_path,
                                        self.configs['result_path'],
                                        f"{self.model_tag}/")
        self.snapshot_path = os.path.join(self.root_path,
                                          self.result_path,
                                          self.configs['snapshot_path'])
        self.device: str = self.configs['device']
        self.mixed_precision: bool = self.configs['mixed_precision']
        if self.mixed_precision:
            # Needed for gradient scaling
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            self.scaler = torch.cuda.amp.GradScaler()

        # Data Parallelism
        self.dp = self.configs['dp']

        # Distributed Data Parallelism Configurations
        self.ddp: bool = self.configs['ddp']['enabled']
        self.node: int = \
            self.configs['ddp']['node'] if self.ddp else 0
        self.local_rank: int = \
            self.configs['ddp']['local_rank'] if self.ddp else 0
        self.rank: int = \
            self.configs['ddp']['rank'] if self.ddp else 0
        self.local_size: int = \
            self.configs['ddp']['local_size'] if self.ddp else 1
        self.world_size: int = \
            self.configs['ddp']['world_size'] if self.ddp else 1

        self.visualization: bool = \
            self.configs['visualization']['enabled']
        self.visualization_chance: float = \
            self.configs['visualization']['chance']
        self.visualization_path = \
            os.path.join(self.root_path,
                         self.result_path,
                         self.configs['visualization']['path'])

        self.visualizer = VisualizerUnet3D(
                _generate_tif=self.configs['visualization']['tif'],
                _generate_gif=self.configs['visualization']['gif'],
                _generate_mesh=self.configs['visualization']['mesh'])

        self.tensorboard: bool = \
            self.configs['tensorboard']['enabled']
        self.tensorboard_path = \
            Path(os.path.join(self.root_path,
                              self.result_path,
                              self.configs['tensorboard']['path']))
        self.tensorboard_path.mkdir(parents=True, exist_ok=True)

        self.skip_training = self.configs['skip_training']

        if self.device == 'cuda':
            self.device_id: int = self.local_rank % torch.cuda.device_count()
        else:
            self.device_id = self.device

        if self.snapshot_path is not None:
            create_dirs_recursively(self.snapshot_path)

        if self.ddp:
            init_process_group(backend="nccl")

        self._prepare_profiling()
        self._prepare_data()

    def __del__(self):
        if self.ddp:
            destroy_process_group()

    def train(self):
        for epoch in range(self.epoch_resume, self.epochs):
            self._train_epoch(epoch)
            # I should later use validation metrics to
            # decide whether overwite to the snapshop or not
            if (epoch + 1) % self.save_interval == 0:
                self._save_sanpshot(epoch)

    # Channels list in the configuration determine
    # Which channels to be stacked
    def _stack_channels(self,
                        _nephrin: Tensor,
                        _wga: Tensor,
                        _collagen4: Tensor) -> Tensor:

        channels: list = self.configs['model']['channels']

        if len(channels) == 3:
            result = torch.cat((_nephrin, _wga, _collagen4), dim=1)
        elif len(channels) == 2:
            if channels in ([0, 1], [1, 0]):
                result = torch.cat((_nephrin, _wga), dim=1)
            if channels in ([0, 2], [2, 0]):
                result = torch.cat((_nephrin, _collagen4), dim=1)
            if channels in ([1, 2], [2, 1]):
                result = torch.cat((_wga, _collagen4), dim=1)
        elif len(channels) == 1:
            # Nephrin -> 0, WGA -> 1, Collagen4 -> 2
            if channels[0] == 0:
                result = _nephrin
            if channels[0] == 1:
                result = _wga
            if channels[0] == 2:
                result = _collagen4
        else:
            assert True, "Wrong channels list configuration"

        return result

    def _log_tensorboard_metrics(self,
                                 _n_iter: int,
                                 _train_accuracy: float,
                                 _train_loss: float,
                                 _valid_accuracy: float,
                                 _valid_loss: float) -> None:
        if not self.tensorboard:
            return
        if not self.configs['tensorboard']['metrics']:
            return
        if self.ddp and self.rank != 0:
            return

        tb_writer = SummaryWriter(self.tensorboard_path.resolve())

        tb_writer.add_scalar('Accuracy/train', _train_accuracy, _n_iter)
        tb_writer.add_scalar('Loss/train', _train_loss, _n_iter)

        tb_writer.add_scalar('Accuracy/valid', _valid_accuracy, _n_iter)
        tb_writer.add_scalar('Loss/valid', _valid_loss, _n_iter)

        tb_writer.close()

    def _visualize_validation(self,
                              _epoch_id: int,
                              _batch_id: int,
                              _inputs,
                              _labels,
                              _predictions,
                              _all: bool = False):

        if not self.visualization:
            return

        random.seed(datetime.now().timestamp())
        dice = random.random()
        if dice > self.visualization_chance:
            return

        batch_size = len(_inputs)
        sample_id = random.randint(0, batch_size-1)

        if self.ddp:
            base_path: str = self.visualization_path + \
                             f"/worker-{self.rank:02}/epoch-{_epoch_id}" + \
                             f"/batch-{_batch_id}/"
        else:
            base_path: str = self.visualization_path + \
                             f"/epoch-{_epoch_id}/batch-{_batch_id}/"

        if _all:
            for index in range(batch_size):
                path: Path = Path(f"{base_path}{index}/")
                path.mkdir(parents=True, exist_ok=True)
                output_dir: str = path.resolve()
                self.visualizer.draw_channels(_inputs[index],
                                              output_dir,
                                              _multiplier=127)
                self.visualizer.draw_labels(_labels[index],
                                            output_dir,
                                            _multiplier=127)
                self.visualizer.draw_predictions(_predictions[index],
                                                 output_dir,
                                                 _multiplier=255)
        else:
            path: Path = Path(f"{base_path}")
            path.mkdir(parents=True, exist_ok=True)
            output_dir: str = path.resolve()

            self.visualizer.draw_channels(_inputs[sample_id],
                                          output_dir)
            self.visualizer.draw_labels(_labels[sample_id],
                                        output_dir,
                                        _multiplier=127)
            self.visualizer.draw_predictions(_predictions[sample_id],
                                             output_dir,
                                             _multiplier=127)

    def _save_sanpshot(self, epoch: int) -> None:
        if self.snapshot_path is None:
            return
        if self.rank > 0:
            return

        snapshot = {}
        snapshot['EPOCHS'] = epoch
        if self.ddp:
            snapshot['MODEL_STATE'] = self.model.module.state_dict()
        else:
            snapshot['MODEL_STATE'] = self.model.state_dict()

        save_path = \
            os.path.join(self.snapshot_path,
                         f"{self.model_tag}-{epoch:03d}.pt")
        torch.save(snapshot, save_path)
        logging.info("Snapshot saved on epoch %d", epoch)

    def _load_snapshot(self) -> None:
        if not os.path.exists(self.snapshot_path):
            return

        snapshot_list = sorted(filter(os.path.isfile,
                                      glob.glob(self.snapshot_path + '*')),
                               reverse=True)

        if len(snapshot_list) <= 0:
            return

        load_path = snapshot_list[0]

        snapshot = torch.load(load_path,
                              map_location=torch.device(self.device_id))

        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epoch_resume = snapshot['EPOCHS'] + 1
        logging.info("Resuming training at epoch: %d", self.epoch_resume)

    def _prepare_data(self) -> None:

        training_ds_dir: str = os.path.join(self.root_path,
                                            self.configs['train_ds']['path'])
        training_sample_dimension: list = \
            self.configs['train_ds']['sample_dimension']
        training_pixel_stride: list = \
            self.configs['train_ds']['pixel_stride']
        training_channel_map: list = \
            self.configs['train_ds']['channel_map']
        training_dataset = GBMDataset(
                _source_directory=training_ds_dir,
                _sample_dimension=training_sample_dimension,
                _pixel_per_step=training_pixel_stride,
                _channel_map=training_channel_map,
                _ignore_stride_mismatch=self.configs['train_ds'][
                    'ignore_stride_mismatch'],
                _label_correction_function=self.label_correction)

        self.number_class = training_dataset.get_number_of_classes()

        validation_ds_dir: str = os.path.join(self.root_path,
                                              self.configs['valid_ds']['path'])
        validation_sample_dimension: list = \
            self.configs['valid_ds']['sample_dimension']
        validation_pixel_stride: list = \
            self.configs['valid_ds']['pixel_stride']
        validation_channel_map: list = \
            self.configs['valid_ds']['channel_map']
        validation_dataset = GBMDataset(
                _source_directory=validation_ds_dir,
                _sample_dimension=validation_sample_dimension,
                _pixel_per_step=validation_pixel_stride,
                _channel_map=validation_channel_map,
                _ignore_stride_mismatch=self.configs['train_ds'][
                    'ignore_stride_mismatch'],
                _label_correction_function=self.label_correction)

        if self.ddp:
            train_sampler = DistributedSampler(training_dataset)
            valid_sampler = DistributedSampler(validation_dataset)
        else:
            train_sampler = None
            valid_sampler = None

        training_batch_size: int = self.configs['train_ds']['batch_size']
        training_shuffle: bool = self.configs['train_ds']['shuffle']
        self.training_loader = DataLoader(training_dataset,
                                          batch_size=training_batch_size,
                                          shuffle=training_shuffle,
                                          sampler=train_sampler,
                                          num_workers=self.configs
                                          ['train_ds']['workers'])

        validation_batch_size: int = self.configs['valid_ds']['batch_size']
        validation_shuffle: bool = self.configs['valid_ds']['shuffle']
        self.validation_loader = DataLoader(validation_dataset,
                                            batch_size=validation_batch_size,
                                            shuffle=validation_shuffle,
                                            sampler=valid_sampler,
                                            num_workers=self.configs
                                            ['valid_ds']['workers'])

    def _prepare_optimizer(self) -> None:
        optimizer_name: str = self.configs['optim']['name']
        if optimizer_name == 'adam':
            lr: float = self.configs['optim']['lr']
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=lr)

    def _prepare_loss(self) -> None:
        if self.ddp:
            device = self.device_id
        else:
            device = self.device

        weights = torch.tensor(self.configs['loss_weights']).to(device)

        if self.configs['mode'] == 'supervised':

            loss_name: str = self.configs['loss']
            if loss_name == 'DiceLoss':
                self.loss = DiceLoss(_weight=weights)
            if loss_name == 'CrossEntropy':
                self.loss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss = SelfSupervisedLoss(self.epochs,
                                           weights)

    def _prepare_profiling(self) -> None:
        if self.configs['profiling']['enabled']:
            scheduler_wait = self.configs['profiling']['scheduler']['wait']
            scheduler_warmup = self.configs['profiling']['scheduler']['warmup']
            scheduler_active = self.configs['profiling']['scheduler']['active']
            scheduler_repeat = self.configs['profiling']['scheduler']['repeat']

            save_path = os.path.join(self.root_path,
                                     self.result_path,
                                     self.configs['profiling']['path'])

            profile_memory = self.configs['profiling']['profile_memory']
            record_shapes = self.configs['profiling']['record_shapes']
            with_flops = self.configs['profiling']['with_flops']
            with_stack = self.configs['profiling']['with_stack']

            tb_trace_handler = \
                torch.profiler.tensorboard_trace_handler(save_path)

            trace_file = os.path.join(save_path,
                                      f"{self.model_tag}.txt")

            def txt_trace_handler(prof):
                with open(trace_file, 'w') as file:
                    file.write(prof.key_averages().table(
                                    sort_by="self_cuda_time_total",
                                    row_limit=-1))

            def print_trace_handler(prof):
                print(prof.key_averages().table(
                                sort_by="self_cuda_time_total",
                                row_limit=-1))

            def trace_handler(prof):
                if self.configs['profiling']['save']['tensorboard']:
                    tb_trace_handler(prof)

                if self.configs['profiling']['save']['text']:
                    txt_trace_handler(prof)

                if self.configs['profiling']['save']['print']:
                    print_trace_handler(prof)

            self.pytorch_profiling = True
            self.prof = torch.profiler.profile(
                    activities=[ProfilerActivity.CUDA,
                                ProfilerActivity.CPU],
                    schedule=torch.profiler.schedule(wait=scheduler_wait,
                                                     warmup=scheduler_warmup,
                                                     active=scheduler_active,
                                                     repeat=scheduler_repeat),
                    on_trace_ready=trace_handler,
                    profile_memory=profile_memory,
                    record_shapes=record_shapes,
                    with_flops=with_flops,
                    with_stack=with_stack)
        else:
            self.pytorch_profiling = False

    @abstractmethod
    def _training_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> (dict, dict):
        pass

    @abstractmethod
    def _validate_step(self,
                       _epoch_id: int,
                       _batch_id: int,
                       _data: dict) -> (dict, dict):
        pass

    @abstractmethod
    def _train_epoch(self, _epoch: int) -> None:
        pass
