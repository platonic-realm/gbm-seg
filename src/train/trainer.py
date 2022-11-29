"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imports
# Python's wierd implementation of abstract methods
from abc import ABC, abstractmethod
from pathlib import Path

# Libary Imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group, destroy_process_group

# Local Imports
from src.utils.misc import create_dirs_recursively


# Tip for using abstract methods in python... dont use
# double __ for the abstract method as python name
# mangeling will mess them and you are going to have a hard time
class Trainer(ABC):
    def __init__(self, _configs: dict):
        self.configs: dict = _configs['trainer']
        self.tqdm: bool = _configs['logging']['tqdm']

        # Note we are using self.config here ...
        self.epochs: int = self.configs['epochs']
        self.epoch_resume = 0
        self.save_interval = self.configs['save_interval']
        self.snapshot_path = self.configs['snapshot_path']
        self.device: str = self.configs['device']
        self.mixed_precision: bool = self.configs['mixed_precision']

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

        self.tensorboard: bool = \
            self.configs['tensorboard']['enabled']
        self.tensorboard_path = \
            Path(self.configs['tensorboard']['path'])
        self.tensorboard_path.mkdir(parents=True, exist_ok=True)

        if self.device == 'cuda':
            self.device_id: int = self.local_rank % torch.cuda.device_count()

        if self.snapshot_path is not None:
            create_dirs_recursively(self.snapshot_path)

        if self.ddp:
            init_process_group(backend="nccl")

    def __del__(self):
        if self.ddp:
            destroy_process_group()

    def train(self):
        for epoch in range(self.epoch_resume, self.epochs):
            self._train_epoch(epoch)
            # I should later use validation metrics to
            # decide whether overwite to the snapshop or not
            if epoch % self.save_interval == 0:
                self._save_sanpshot(epoch)

    @abstractmethod
    def _save_sanpshot(self, epoch: int) -> None:
        pass

    @abstractmethod
    def _load_snapshot(self) -> None:
        pass

    @abstractmethod
    def _prepare_data(self) -> None:
        pass

    @abstractmethod
    def _prepare_optimizer(self) -> None:
        pass

    @abstractmethod
    def _prepare_loss(self) -> None:
        pass

    @abstractmethod
    def _training_step(self, _data: dict) -> (dict, dict):
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
