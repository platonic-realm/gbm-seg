"""Model registry for the gbm-seg factory.

Maps the ``configs.trainer.model.name`` config string to a build callable
that returns an ``nn.Module`` satisfying the project's model interface
contract (post-A3 refactor):

    forward(x: Tensor) -> (logits: Tensor, outputs: Tensor)

The model is stateless w.r.t. inference; the sliding-window accumulator
lives in :mod:`src.infer.stitching`.

Each model's build callable takes ``(configs, input_channels, num_classes)``
and is responsible for translating its own slice of ``configs.trainer.model.*``
into constructor arguments. Add a new model by writing its module and
registering it below — the rest of the codebase (factory, trainer, inferer)
needs no changes.
"""

from typing import Callable

from src.models.swin_unetr import build as _build_swin_unetr
from src.models.unet3d import build as _build_unet3d

MODEL_REGISTRY: dict[str, Callable] = {
    'unet_3d': _build_unet3d,
    'swin_unetr': _build_swin_unetr,
}


def build_model(name: str, configs: dict, input_channels: int, num_classes: int):
    """Construct a segmentation model from the registry.

    Raises ``NotImplementedError`` (not ``KeyError``) on an unknown name,
    so the failure mode is explicit and matches the prior factory
    behaviour.
    """
    if name not in MODEL_REGISTRY:
        raise NotImplementedError(
            f"Unknown model: {name!r}. "
            f"Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](configs, input_channels, num_classes)


__all__ = ['MODEL_REGISTRY', 'build_model']
