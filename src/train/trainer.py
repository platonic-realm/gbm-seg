"""
Author: Arash Fatehi
Date:   30.10.2022
File:   trainer.py
"""

# Python Imports
import os
import sys
import logging

# Library Imports
from torch import nn
from tqdm import tqdm

# Local Imports

class Trainer():
    def __init__(self,
                 _model,
                 _training_function=None,
                 _validation_function=None,
                 _training_dataloader=None,
                 _validation_dataloader=None,
                 _validation_no_of_batches=-1,
                 _optimizer=None,
                 _loss_function=None,
                 _load_from_checkpoint=False,
                 _initialization_method=None,
                 _create_chekpoints=False,
                 _checkpoints_path=None,
                 _tensorboard_support=False,
                 _tensorboard_path=None,
                 _device='cpu',
                 _log_to_file=False,
                 _log_file_path=None,
                 _multi_node_training=False,
                 _node_id=None,
                 _data_parallelism=False,
                 _number_of_gpus=0):

        self.model = _model
        self.training_step = _training_function
        self.validation_step = _validation_function
        self.training_dataloader = _training_dataloader
        self.valiation_dataloader = _validation_dataloader
        self.validation_no_of_batches = _validation_no_of_batches
        self.optimizer = _optimizer
        self.loss_function = _loss_function
        self.load_from_checkpoint = _load_from_checkpoint
        self.initialization_method = _initialization_method
        self.create_checkpoints = _create_chekpoints
        self.checkpoints_path = _checkpoints_path
        self.tensorboard_support = _tensorboard_support
        self.tensorboard_path = _tensorboard_path
        self.device = _device
        self.log_to_file = _log_to_file
        self.log_file_path = _log_file_path
        self.multi_node_training = _multi_node_training
        self.node_id = _node_id
        self.data_parallelism = _data_parallelism
        self.number_of_gpus = _number_of_gpus

        if self.load_from_checkpoint:
            assert self.checkpoints_path is None, \
                "checkpoint_path can not be None if you " \
                "want to load the model from a checkpoint."
            assert not os.path.exists(self.checkpoints_path), \
                "checkpoint file doesn't exist."
            # ToDo: Load the checkpoint
        else:
            # ToDo: Use initialization method
            pass

        if self.create_checkpoints:
            assert self.checkpoints_path is None, \
                "checkpoint_path can not be None if you " \
                "want to save the model's checkpoints"

        if self.tensorboard_support:
            assert self.tensorboard_path is None, \
                "ternsorboard_path can not be None if you " \

        if self.log_to_file:
            # ToDo: add log handler
            pass

        if self.multi_node_training:
            pass

        if self.validation_no_of_batches < 0:
            self.validation_no_of_batches = len(self.valiation_dataloader)

        if self.data_parallelism:
            self.model = nn.DataParallel(self.model)

    def __load_from_file__(self):
        pass

    def __initialize_wights__(self):
        pass

    def validate(self):
        if self.valiation_dataloader is None:
            pass
        if self.validation_step is None:
            pass

        self.model.to(self.device)
        process_bar = tqdm(self.valiation_dataloader,
                           file=sys.stdout,
                           total=self.validation_no_of_batches)

        for index, data in enumerate(process_bar):
            if index >= self.validation_no_of_batches:
                break
            index += 1

            losses, metrics = self.validation_step(self.model,
                                                   data,
                                                   self.device,
                                                   self.loss_function,
                                                   index)

            postfix = {}
            for key in losses:
                postfix[key] = losses[key]
            for key in metrics:
                postfix[key] = metrics[key]

            process_bar.set_description("Validation")
            process_bar.set_postfix(postfix)

    def train(self, _epochs):
        logging.debug("######################")
        logging.debug("Using device: %s", self.device)

        logging.debug("Moving model to device: %s", self.device)
        self.model.to(self.device)

        for epoch in range(_epochs):

            process_bar = tqdm(self.training_dataloader,
                               file=sys.stdout)

            for index, data in enumerate(process_bar):

                losses, metrics = \
                        self.training_step(self.model,
                                           index,
                                           data,
                                           self.device,
                                           self.optimizer,
                                           self.loss_function)

                process_bar.set_description(f"Epoch {epoch+1}/{_epochs}")

                postfix = {}
                for key in losses:
                    postfix[key] = losses[key]
                for key in metrics:
                    postfix[key] = metrics[key]

                process_bar.set_postfix(postfix)

            self.validate()
