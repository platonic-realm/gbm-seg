# Python Imports
import logging


class MetricLogger:
    """Fans per-step metric logging out to all enabled backends.

    Each backend is optional; passing ``None`` for a backend disables it.
    """

    def __init__(self, _metric_wandb=None):
        self.metric_wandb = _metric_wandb

    def log(self,
            _epoch: int,
            _step: int,
            _tag: str,
            _metrics: dict) -> None:

        logging.info("%s, Epoch: %d, Step: %d, Metrics: %s",
                     _tag,
                     _epoch,
                     _step,
                     _metrics)

        if self.metric_wandb is not None:
            self.metric_wandb.log(_epoch, _step, _tag, _metrics)
