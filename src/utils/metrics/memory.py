# Library Imports
import torch

# Local Imports
import src.utils.misc as misc
from src.train.distributed import all_reduce_sum_, is_distributed


class GPURunningMetrics:
    def __init__(self,
                 _device,
                 _metrics):

        self.device = _device

        self.metrics = _metrics

        self.values = torch.zeros(len(self.metrics),
                                  device=self.device,
                                  dtype=torch.float32)
        self.counter = torch.zeros(len(self.metrics),
                                   device=self.device,
                                   dtype=torch.float32)

    def add(self, _values) -> None:
        for i, metric in enumerate(self.metrics):
            if len(_values[metric].shape) > 0:
                _value = _values[metric].squeeze()
            else:
                _value = _values[metric]

            if not torch.isnan(_value):
                self.values[i] += _value
                self.counter[i] += 1

    def calculate(self) -> dict:
        results = dict.fromkeys(self.metrics)

        # Under DDP each rank has accumulated its own shard's numerator
        # (`values`) and denominator (`counter`). Sum-reduce both before
        # dividing so every rank's `.calculate()` returns the same global
        # mean — equivalent to having seen the union of all shards.
        # Note this is an average-of-batch-averages, not a population
        # mean over voxels (which would need confusion-matrix reduce);
        # acceptable for the per-step logging we use.
        if is_distributed():
            all_reduce_sum_(self.values)
            all_reduce_sum_(self.counter)

        results_tensor = torch.div(self.values, self.counter)
        self.values = torch.zeros(len(self.metrics),
                                  device=self.device,
                                  dtype=torch.float32)
        self.counter = torch.zeros(len(self.metrics),
                                   device=self.device,
                                   dtype=torch.float32)

        results_numpy = misc.to_numpy(results_tensor)

        for i, metric in enumerate(self.metrics):
            results[metric] = results_numpy[i]
        return results
