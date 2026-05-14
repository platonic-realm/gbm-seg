"""W&B logging backend for ``MetricLogger``.

Imports ``wandb`` lazily so the rest of the codebase doesn't pay the
~500 ms wandb import cost when wandb logging is disabled. The W&B run
must already be initialised (via ``train.py:maybe_init_wandb``) before
this backend is instantiated — this class only calls ``wandb.log``.

Metrics are namespaced by ``_tag`` (e.g. ``train/Dice``, ``valid/Dice``)
so the W&B UI plots train vs valid curves side by side.
"""

import logging


class MetricWandb:
    """Forwards per-step metrics to the active W&B run.

    Constructor raises ``ImportError`` if wandb is not installed; callers
    should check installability before instantiating.
    """

    def __init__(self):
        import wandb  # lazy
        self.wandb = wandb

    def log(self, _epoch: int, _step: int, _tag: str,
            _metrics: dict) -> None:
        if self.wandb.run is None:
            # No active run; surface as a warning rather than crash so a
            # misconfigured wandb call doesn't take training down.
            logging.warning("MetricWandb.log called with no active wandb run.")
            return

        payload = {f"{_tag}/{name}": _to_python_scalar(value)
                   for name, value in _metrics.items()}
        payload['epoch'] = int(_epoch)
        # Don't pass `step=int(_step)` to wandb.log: wandb 0.26's auto-step
        # counter (incremented by system-metric snapshots over a long run)
        # silently drops explicit step values that fall behind it, which
        # cost us a debugging round. wandb's internal `_step` axis indexes
        # the charts; `epoch` lives in the payload for grouping/filtering.
        # The training-step number is intentionally NOT in the payload —
        # plotting it would create a redundant linear-over-_step chart.
        self.wandb.log(payload)


def _to_python_scalar(value):
    """Coerce torch tensors / numpy scalars to plain Python numbers.

    W&B accepts these natively too, but plain floats are smaller in
    transit and avoid surprises when filtering on the UI side.
    """
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (RuntimeError, ValueError):
            pass
    return float(value)
