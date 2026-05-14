# Python Imports
import logging


class MetricLogger:
    """Fans per-step metric logging out to all enabled backends.

    Each backend is optional; passing ``None`` for a backend disables it.

    Under DDP this method is called once per rank with each rank's
    already-globally-averaged metrics (see ``GPURunningMetrics.calculate``
    which all-reduces internally). To avoid 4 duplicate stdout lines and
    4 duplicate wandb.log calls, the fan-out is gated to rank-0.
    """

    def __init__(self, _metric_wandb=None):
        self.metric_wandb = _metric_wandb

    def log(self,
            _epoch: int,
            _step: int,
            _tag: str,
            _metrics: dict) -> None:
        # Lazy import to avoid forcing distributed init at module-load time.
        from src.train.distributed import is_main_process
        if not is_main_process():
            return

        logging.info("%s, Epoch: %d, Step: %d, Metrics: %s",
                     _tag,
                     _epoch,
                     _step,
                     _metrics)

        if self.metric_wandb is not None:
            self.metric_wandb.log(_epoch, _step, _tag, _metrics)
