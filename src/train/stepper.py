# Python Imports
from abc import ABC, abstractmethod

# Library Imports
import torch
from torch import Tensor
from torch import nn

# Local Imports


class StepperInterface(ABC):

    @abstractmethod
    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 loss_function):
        pass

    @abstractmethod
    def step(self,
             sample: Tensor,
             labels: Tensor) -> (Tensor, Tensor, Tensor):
        pass

    @abstractmethod
    def getSteps(self) -> int:
        pass

    @abstractmethod
    def getSeenLabels(self) -> int:
        pass


class StepperSimple(StepperInterface):

    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.steps = 0
        self.seen_labels = 0

    def step(self,
             sample: Tensor,
             labels: Tensor) -> (Tensor, Tensor, Tensor):

        self.optimizer.zero_grad()

        logits, results = self.model(sample)
        loss = self.loss_function(logits, labels)
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        self.seen_labels += sample.shape[0]

        return logits, results, loss

    def getSteps(self) -> int:
        return self.steps

    def getSeenLabels(self) -> int:
        return self.seen_labels


class StepperMixedPrecision(StepperInterface):

    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        # Needed for gradient scaling
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        self.scaler = torch.cuda.amp.GradScaler()

        self.steps = 0
        self.seen_labels = 0

    def step(self,
             sample: Tensor,
             labels: Tensor) -> (Tensor, Tensor, Tensor):

        self.optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, results = self.model(sample)
            loss = self.loss_function(logits, labels)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.steps += 1
        self.seen_labels += sample.shape[0]

        return logits, results, loss

    def getSteps(self) -> int:
        return self.steps

    def getSeenLabels(self) -> int:
        return self.seen_labels
