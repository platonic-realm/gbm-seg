# Python Imports
from abc import ABC, abstractmethod

# Library Imports
import torch
from torch import Tensor, nn

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


class StepperSimple(StepperInterface):

    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.steps = 0

    def step(self,
             sample: Tensor,
             labels: Tensor) -> (Tensor, Tensor, Tensor):

        self.optimizer.zero_grad()

        logits, results = self.model(sample)
        loss = self.loss_function(logits, labels)
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return logits, results, loss

    def getSteps(self) -> int:
        return self.steps

    def state_dict(self) -> dict:
        """Picklable state for resume. No scaler in the simple path."""
        return {'steps': self.steps, 'scaler': None}

    def load_state_dict(self, _state: dict) -> None:
        self.steps = int(_state.get('steps', 0))


class StepperMixedPrecision(StepperInterface):

    def __init__(self,
                 model: nn.Module,
                 optimizer,
                 loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        # https://pytorch.org/docs/stable/notes/amp_examples.html
        self.scaler = torch.amp.GradScaler('cuda')

        self.steps = 0

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

        return logits, results, loss

    def getSteps(self) -> int:
        return self.steps

    def state_dict(self) -> dict:
        """Picklable state for resume: step counter + GradScaler state.
        The scaler holds the dynamic loss-scaling factor — restoring it
        avoids the warmup that would otherwise blow the first few resumed
        steps with overflowing gradients."""
        return {'steps': self.steps,
                'scaler': self.scaler.state_dict()}

    def load_state_dict(self, _state: dict) -> None:
        self.steps = int(_state.get('steps', 0))
        scaler_state = _state.get('scaler')
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)
