"""Unet3D forward-pass smoke test (CPU only).

A3 refactor: Unet3D is now stateless w.r.t. inference. Sliding-window
accumulation lives in src/infer/stitching.py:StitchAccumulator (exercised
by tests/infer/test_stitching.py). This file only tests the model
interface contract: `forward(x) -> (logits, outputs)`.
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
    )
    x = torch.randn(1, 3, *sample_dim)
    logits, outputs = model(x)
    # Logits: (B, C, D, H, W)
    assert logits.shape == (1, 2, *sample_dim)
    # Outputs (post-argmax): (B, D, H, W) int
    assert outputs.shape == (1, *sample_dim)
    assert outputs.dtype in (torch.int64, torch.long)


def test_unet3d_no_inference_state():
    """The A3 refactor removed inference-mode state from the model.

    The model must not carry a ``result_tensor`` / ``inference`` /
    ``get_result`` API: those live on ``StitchAccumulator`` now.
    """
    model = Unet3D(
        _name='unet_3d',
        _input_channels=3,
        _number_of_classes=2,
        _feature_maps=(8, 16),
        _sample_dimension=[4, 16, 16],
    )
    assert not hasattr(model, 'result_tensor')
    assert not hasattr(model, 'inference')
    assert not hasattr(model, 'get_result')
