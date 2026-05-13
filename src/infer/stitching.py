"""Sliding-window inference stitching.

Aggregates per-patch model outputs into a full-volume segmentation mask.
Four modes, selectable via ``configs.inference.stitching``:

- ``sum_logits``  — legacy: sum raw logits at each patch's offset, argmax once at the end.
- ``gaussian``    — new default (nnU-Net convention): Gaussian window with ``sigma = patch_size/8`` applied to softmax probabilities; divide weighted-sum by weight at the end.
- ``hann``        — parameter-free; sharper rolloff than Gaussian.
- ``flat_softmax``— uniform window over softmax probabilities (diagnostic baseline; isolates the softmax-vs-logits effect from the window-vs-flat effect).

The four modes factorise the design space:
``softmax-vs-logits × window-vs-flat``.

Moved out of ``Unet3D.forward`` in A3 (see plan file) so that the model
interface is minimal — just ``forward(x) → (logits, outputs)``. Future
models (e.g. custom SwinUNETR for shallow Z-stacks) don't have to
re-implement the accumulator.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

# Modes whose contribution is softmax(logits) rather than raw logits.
_MODES_SOFTMAX = {'gaussian', 'hann', 'flat_softmax'}


def _gaussian_window(patch_size: Sequence[int]) -> Tensor:
    """Separable 3D Gaussian peaked at the patch centre, scaled so peak == 1."""
    axes = []
    for size in patch_size:
        sigma = size / 8.0
        centre = (size - 1) / 2.0
        idx = torch.arange(size, dtype=torch.float32)
        axes.append(torch.exp(-0.5 * ((idx - centre) / sigma) ** 2))
    window = axes[0]
    for axis in axes[1:]:
        window = window.unsqueeze(-1) * axis
    return window / window.max()


def _hann_window(patch_size: Sequence[int]) -> Tensor:
    """Separable 3D Hann window."""
    axes = [torch.hann_window(size, periodic=False, dtype=torch.float32)
            for size in patch_size]
    window = axes[0]
    for axis in axes[1:]:
        window = window.unsqueeze(-1) * axis
    return window


def _uniform_window(patch_size: Sequence[int]) -> Tensor:
    return torch.ones(*patch_size, dtype=torch.float32)


_WINDOWS = {
    'sum_logits': _uniform_window,
    'gaussian': _gaussian_window,
    'hann': _hann_window,
    'flat_softmax': _uniform_window,
}

VALID_MODES = tuple(_WINDOWS.keys())


class StitchAccumulator:
    """Full-volume accumulator for patch-based 3D segmentation inference.

    Holds two CPU tensors:
        - ``numerator`` : shape ``(C, *result_shape[1:])``, accumulates
          ``window * contribution`` over all patches.
        - ``weight``    : shape ``result_shape[1:]``, accumulates the
          window itself (so we can divide by it at the end).

    ``contribution`` is either raw logits (``sum_logits``) or softmax
    probabilities (every other mode). ``finalize()`` divides numerator
    by weight (skipping division for ``sum_logits`` since it has no
    meaningful softmax interpretation), then argmaxes over the class
    dimension to produce the per-voxel class mask.

    Offset convention matches the data pipeline (``src/data/ds_infer.py``):
    ``offsets[b]`` is a 4-tuple where indices ``[1], [2], [3]`` are
    ``x_start, y_start, z_start`` respectively.
    """

    def __init__(self,
                 mode: str,
                 patch_size: Sequence[int],
                 result_shape: Sequence[int]):
        if mode not in _WINDOWS:
            raise ValueError(
                f"Unknown stitching mode {mode!r}; "
                f"valid options: {VALID_MODES}")
        if len(patch_size) != 3:
            raise ValueError(f"patch_size must be 3D (Z, X, Y); got {patch_size}")
        if len(result_shape) != 4:
            raise ValueError(
                f"result_shape must be 4D (C, Z, X, Y); got {result_shape}")

        self.mode = mode
        self.patch_size = tuple(int(s) for s in patch_size)
        self.result_shape = tuple(int(s) for s in result_shape)

        self.window = _WINDOWS[mode](self.patch_size)
        self.numerator = torch.zeros(self.result_shape, dtype=torch.float32)
        self.weight = torch.zeros(self.result_shape[1:], dtype=torch.float32)

    def add_batch(self, logits: Tensor, offsets: Tensor) -> None:
        """Accumulate one batch of patch logits at their offsets.

        Args:
            logits:  ``(B, C, Z, X, Y)`` — model output for the batch.
            offsets: ``(B, 4)`` — per-row offsets; indices ``[1], [2], [3]``
                are ``x_start, y_start, z_start``.
        """
        if self.mode in _MODES_SOFTMAX:
            contribution = torch.softmax(logits, dim=1).to('cpu')
        else:
            contribution = logits.to('cpu')

        offsets_cpu = offsets.to('cpu')
        Z, X, Y = self.patch_size
        for batch_id in range(offsets_cpu.shape[0]):
            x_start = int(offsets_cpu[batch_id][1])
            y_start = int(offsets_cpu[batch_id][2])
            z_start = int(offsets_cpu[batch_id][3])
            spatial = (slice(z_start, z_start + Z),
                       slice(x_start, x_start + X),
                       slice(y_start, y_start + Y))
            full = (slice(None),) + spatial
            self.numerator[full] += self.window * contribution[batch_id]
            self.weight[spatial] += self.window

    def finalize(self) -> Tensor:
        """Per-voxel argmax mask of shape ``result_shape[1:]``."""
        if self.mode == 'sum_logits':
            # Legacy semantics: no division, argmax raw summed logits.
            return torch.argmax(self.numerator, dim=0)
        eps = 1e-8
        final = self.numerator / self.weight.clamp_min(eps).unsqueeze(0)
        return torch.argmax(final, dim=0)
