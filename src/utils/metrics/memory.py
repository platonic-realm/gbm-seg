# Library Imports
import torch

# Local Imports
import src.utils.misc as misc


class CPURunningMetric():
    def __init__(self):
        self.value: float = 0
        self.counter: int = 0

    def add(self, _value: any) -> None:
        self.value += float(_value)
        self.counter += 1

    def calcualte(self) -> float:
        result: float = self.value / self.counter
        self.value = 0.0
        self.counter = 0
        return result


class GPURunningMetrics():
    def __init__(self,
                 _device,
                 _metrics):

        self.device = _device

        self.metrics = _metrics

        self.values = torch.zeros(len(self.metrics),
                                  device=self.device,
                                  dtype=torch.float32)
        self.counter = torch.zeros(1,
                                   device=self.device,
                                   dtype=torch.float32)

    def add(self, _values) -> None:
        for i, metric in enumerate(self.metrics):
            if len(_values[metric].shape) > 0:
                self.values[i] += _values[metric].squeeze()
            else:
                self.values[i] += _values[metric]
        self.counter += 1

    def calculate(self) -> dict:
        results = dict.fromkeys(self.metrics)
        results_tensor = torch.div(self.values, self.counter)
        self.values = torch.zeros(len(self.metrics),
                                  device=self.device,
                                  dtype=torch.float32)
        self.counter = torch.zeros(1,
                                   device=self.device,
                                   dtype=torch.float32)

        results_numpy = misc.to_numpy(results_tensor)

        for i, metric in enumerate(self.metrics):
            results[metric] = results_numpy[i]
        return results
