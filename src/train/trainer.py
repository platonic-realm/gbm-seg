"""
Author: Arash Fatehi
Date:   22.11.2022
"""

# Python Imports
# Python's wierd implementation of abstract methods
from abc import ABC, abstractmethod

# Libary Imports

# Local Imports


class Trainer(ABC):
    def __init__(self, _configs: dict):
        self.configs: dict = _configs['trainer']
        self.tqdm: bool = _configs['logging']['tqdm']

        # Note we are using self.config here ...
        self.epochs: int = self.configs['epochs']
        self.epoch_resume = 0
        self.save_interval = self.configs['save_interval']
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

    def train(self):
        for epoch in range(self.epoch_resume, self.epochs):
            _, _ = self.__training_step(epoch)
            _, _ = self.__validate_step(epoch)
            # I should later use validation metrics to
            # decide whether overwite to the snapshop or not
            if epoch % self.save_interval == 0:
                self.__save_sanpshot()

    @abstractmethod
    def __save_sanpshot(self) -> None:
        pass

    @abstractmethod
    def __load_snapshot(self) -> None:
        pass

    @abstractmethod
    def __log_tensorboard(self) -> None:
        pass

    @abstractmethod
    def __prepare_data(self) -> None:
        pass

    @abstractmethod
    def __prepare_optimizer(self) -> None:
        pass

    @abstractmethod
    def __prepare_loss(self) -> None:
        pass

    @abstractmethod
    def __training_step(self, epoch: int) -> (dict, dict):
        pass

    @abstractmethod
    def __validate_step(self, epoch: int) -> (dict, dict):
        pass
