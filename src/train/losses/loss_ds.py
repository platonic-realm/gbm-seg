"""Deep supervision loss wrapper.

Wraps any base segmentation loss so that, when the model produces a list
of logits at multiple decoder resolutions, each is supervised against a
downsampled copy of the labels and the per-level losses are summed with
geometrically-decaying weights.

Standard pattern (Isensee et al. nnU-Net, *Nat Methods* 2021): weights are
``0.5 ** level`` for the deepest ``L`` decoder heads; normalised to sum
to 1 so the overall loss magnitude is unaffected by the number of heads.

When the model produces a single tensor (deep supervision disabled at
the model level), the wrapper degrades transparently to the base loss.
"""

from collections.abc import Sequence
from typing import Union

import torch.nn.functional as F
from torch import Tensor, nn


def default_ds_weights(num_heads: int) -> list[float]:
    """``[1, 0.5, 0.25, ...]`` normalised to sum to 1."""
    raw = [0.5 ** k for k in range(num_heads)]
    total = sum(raw)
    return [w / total for w in raw]


class DeepSupervisionLoss(nn.Module):
    """Wrap a base segmentation loss with deep-supervision combination.

    When the model produces a list of logits, applies ``base_loss`` at
    every level against a correspondingly-downsampled label tensor.
    Labels are downsampled with ``mode='nearest'`` to preserve integer
    class identity.

    When the model produces a single tensor (DS disabled), forwards to
    ``base_loss`` unchanged.
    """

    def __init__(self,
                 base_loss: nn.Module,
                 weights: Sequence[float] = None):
        super().__init__()
        self.base_loss = base_loss
        # ``weights`` is stored as a list of plain floats so it doesn't
        # show up in self.parameters() (nothing here is learnable).
        self.weights = list(weights) if weights is not None else None

    def forward(self,
                logits: Union[Tensor, list[Tensor]],
                labels: Tensor) -> Tensor:
        if not isinstance(logits, (list, tuple)):
            return self.base_loss(logits, labels)

        weights = self.weights or default_ds_weights(len(logits))
        if len(weights) != len(logits):
            raise ValueError(
                f"DS weight count ({len(weights)}) must match the number of "
                f"logit tensors ({len(logits)}).")

        total = None
        for logits_k, w in zip(logits, weights):
            labels_k = _downsample_labels(labels, target_shape=logits_k.shape[2:])
            term = w * self.base_loss(logits_k, labels_k)
            total = term if total is None else total + term
        return total


def _downsample_labels(labels: Tensor, target_shape) -> Tensor:
    """Nearest-neighbour downsample integer labels to ``target_shape``.

    Labels have shape ``(B, Z, X, Y)``; ``F.interpolate`` expects a
    channel dim, so we unsqueeze, cast to float for interpolation, then
    cast back to the original integer dtype.
    """
    if tuple(labels.shape[1:]) == tuple(target_shape):
        return labels  # no-op fast path for the final-resolution head
    return F.interpolate(
        labels.unsqueeze(1).float(),
        size=tuple(target_shape),
        mode='nearest',
    ).squeeze(1).to(labels.dtype)
