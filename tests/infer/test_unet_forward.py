"""Unet3D forward-pass smoke test (CPU only).

Builds a tiny Unet3D, pushes a synthetic batch through it, and verifies
that (a) shapes match expectations in training mode and (b) inference
mode accumulates logits at the per-batch offsets into result_tensor.
"""

import torch

from src.models.unet3d.unet3d import Unet3D


def test_unet3d_forward_training_shapes():
    sample_dim = [4, 16, 16]
    model = Unet3D(
        _name='unet_3d',
        _input_channels=3,
        _number_of_classes=2,
        _feature_maps=(8, 16),
        _sample_dimension=sample_dim,
        _inference=False,
    )
    x = torch.randn(1, 3, *sample_dim)
    logits, outputs = model(x)
    # Logits: (B, C, D, H, W)
    assert logits.shape == (1, 2, *sample_dim)
    # Outputs (post-argmax): (B, D, H, W)
    assert outputs.shape == (1, *sample_dim)
    assert outputs.dtype == torch.int64 or outputs.dtype == torch.long


def test_unet3d_inference_accumulates_at_offsets():
    sample_dim = [4, 16, 16]
    result_shape = (2, 8, 32, 32)  # (C, D, H, W)
    model = Unet3D(
        _name='unet_3d',
        _input_channels=3,
        _number_of_classes=2,
        _feature_maps=(8, 16),
        _sample_dimension=sample_dim,
        _inference=True,
        _result_shape=list(result_shape),
    )
    model.eval()

    x = torch.randn(2, 3, *sample_dim)
    # Per unet3d.py:120-132 the indexing per batch row is
    #   x_start = offsets[b][1], y_start = offsets[b][2], z_start = offsets[b][3]
    # and the write is result[:, z:z+D, x:x+X, y:y+Y].
    # So row=[_, x_start, y_start, z_start].
    offsets = torch.tensor([
        [0,  0, 0, 0],   # writes to result[:, 0:4,  0:16,  0:16]
        [1, 16, 0, 4],   # writes to result[:, 4:8, 16:32, 0:16]
    ], dtype=torch.long)

    with torch.no_grad():
        model(x, offsets)

    result = model.get_result()
    assert result.shape == result_shape
    assert result[:, 0:4, 0:16, 0:16].abs().sum() > 0
    assert result[:, 4:8, 16:32, 0:16].abs().sum() > 0
