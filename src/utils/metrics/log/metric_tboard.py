# Python Imports
from pathlib import Path

# Library Imports
from torch.utils.tensorboard import SummaryWriter

# Local Imports


class MetricTensorboard():

    def __init__(self,
                 _tensorboard_path: str,
                 _zero_metrics: dict,
                 _record_seen_labels: bool):

        self.tensorboard_path = Path(_tensorboard_path)
        self.tensorboard_path.mkdir(parents=True, exist_ok=True)

        self.log(0, 'train', _zero_metrics)
        self.log(0, 'valid', _zero_metrics)

        if _record_seen_labels:
            self.log(0, 'train_sl', _zero_metrics)
            self.log(0, 'valid_sl', _zero_metrics)

    def log(self,
            _interval,
            _tag,
            _metrics) -> None:

        tb_writer = SummaryWriter(self.tensorboard_path.resolve())

        for metric in _metrics.keys():
            tb_writer.add_scalar(f'{metric}/{_tag}',
                                 _metrics[metric],
                                 _interval)

        tb_writer.close()
