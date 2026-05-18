"""Unet3D forward-pass smoke test (CPU only).

A3 refactor: Unet3D is now stateless w.r.t. inference. Sliding-window
accumulation lives in src/infer/stitching.py:StitchAccumulator (exercised
by tests/infer/test_stitching.py). This file only tests the model
interface contract: `forward(x) -> (logits, outputs)`.
"""

import torch

from src.models.unet3d.unet3d import Unet3D


def test_unet3d_forward_training_shapes():
    # Need ≥3 encoder stages so the last decoder layer upsamples back to E0
    # and the output Z matches the input Z. The Z-deduct pooling makes the
    # 2-stage case degenerate (output Z = bottleneck Z), so use 3 stages.
    sample_dim = [12, 16, 16]
    model = Unet3D(
        _name='unet_3d',
        _input_channels=3,
        _number_of_classes=2,
        _feature_maps=(8, 16, 32),
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
        _feature_maps=(8, 16, 32),
        _sample_dimension=[12, 16, 16],
    )
    assert not hasattr(model, 'result_tensor')
    assert not hasattr(model, 'inference')
    assert not hasattr(model, 'get_result')


def test_unet3d_auto_z_deduction_scales_with_patch_depth():
    """'auto' z_deduction derives from the Z patch depth — the encoder
    reduces Z to ~half its depth over the num_stages-1 pools; an explicit
    int overrides. Same anisotropic rule as SwinUNETR."""
    # 4 stages, Z=12 -> round(12/(2*3)) = 2 (the historical default).
    m = Unet3D(_name='unet_3d', _input_channels=3, _number_of_classes=2,
               _feature_maps=(8, 16, 32, 64), _sample_dimension=[12, 32, 32])
    assert m.z_deduction == 2
    # 4 stages, Z=24 -> round(24/6) = 4.
    m = Unet3D(_name='unet_3d', _input_channels=3, _number_of_classes=2,
               _feature_maps=(8, 16, 32, 64), _sample_dimension=[24, 64, 64])
    assert m.z_deduction == 4
    # An explicit int overrides 'auto'.
    m = Unet3D(_name='unet_3d', _input_channels=3, _number_of_classes=2,
               _feature_maps=(8, 16, 32, 64), _sample_dimension=[24, 64, 64],
               _z_deduction_per_stage=2)
    assert m.z_deduction == 2


def test_unet3d_checkpointing_setting_resolves():
    """'on'/'off'/bool/None resolve deterministically; 'auto' is off on CPU."""
    resolve = Unet3D._resolve_checkpointing
    assert resolve('on') is True
    assert resolve(True) is True
    assert resolve('off') is False
    assert resolve(False) is False
    assert resolve(None) is False
    assert resolve('auto') is False   # no CUDA device in the test env


def test_unet3d_forward_backward_with_checkpointing():
    """With checkpointing forced on, forward + backward still run and
    produce finite gradients (the checkpointed layers are exact)."""
    sample_dim = [12, 16, 16]
    model = Unet3D(
        _name='unet_3d',
        _input_channels=3,
        _number_of_classes=2,
        _feature_maps=(8, 16, 32),
        _sample_dimension=sample_dim,
        _gradient_checkpointing='on',
    )
    assert model.use_checkpointing is True
    x = torch.randn(1, 3, *sample_dim, requires_grad=True)
    logits, _ = model(x)
    assert logits.shape == (1, 2, *sample_dim)
    logits.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_unet3d_checkpointing_matches_plain_forward():
    """Checkpointing is exact: with grad enabled (so the checkpoint path is
    actually taken) the logits match the un-checkpointed forward."""
    sample_dim = [12, 16, 16]
    kwargs = dict(_name='unet_3d', _input_channels=3, _number_of_classes=2,
                  _feature_maps=(8, 16, 32), _sample_dimension=sample_dim)
    plain = Unet3D(**kwargs, _gradient_checkpointing='off')
    ckpt = Unet3D(**kwargs, _gradient_checkpointing='on')
    ckpt.load_state_dict(plain.state_dict())
    x = torch.randn(1, 3, *sample_dim)
    # Grad is enabled by default, so `ckpt` takes the checkpoint() branch.
    logits_plain, _ = plain(x)
    logits_ckpt, _ = ckpt(x)
    assert torch.allclose(logits_plain, logits_ckpt, atol=1e-5)
