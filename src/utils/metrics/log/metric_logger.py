# Python Imports
import logging

# Library Imports

# Local Imports
from src.utils.metrics.log.metric_sql import MetricSQL
from src.utils.metrics.log.metric_tboard import MetricTensorboard


class MetricLogger():

    def __init__(self,
                 _metric_sql: MetricSQL = None,
                 _metric_tboard: MetricTensorboard = None):
        self.metric_sql: MetricSQL = _metric_sql
        self.metric_tboard: MetricTensorboard = _metric_tboard

    def log(self,
            _epoch: int,
            _step: int,
            _seen_labels,
            _tag: str,
            _metrics: dict) -> None:

        logging.info("%s, Epoch: %d, Step: %d, Seen Labels: %d, Metrics: %s",
                     _tag,
                     _epoch,
                     _step,
                     _seen_labels,
                     _metrics)

        if self.metric_sql is not None:
            self.metric_sql.log(_epoch, _step, _seen_labels, _tag, _metrics)

        if self.metric_tboard is not None:
            self.metric_tboard.log(_interval=_step, _tag=_tag, _metrics=_metrics)
