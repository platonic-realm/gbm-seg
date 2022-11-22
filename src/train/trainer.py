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
    def __init__(self):
        pass

    def __save_sanpshot(self):
        pass

    def __load_snapshot(self):
        pass

    def __log_tensorboard(self):
        pass

    def __prepare_data(self):
        pass

    def train(self):
        pass

    @abstractmethod
    def __training_step(self):
        pass

    @abstractmethod
    def __validate_step(self):
        pass
