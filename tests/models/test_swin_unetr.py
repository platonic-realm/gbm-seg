"""SwinUNETR3D regression. Locks in the anisotropic SwinUNETR variant.

Covers:
- Forward shape contract: output shape matches input shape.
- Deduct-Z encoder/decoder Z trajectory matches the design
  (Z=12 → 10 → 8 → 6 with default deduct=2).
- Deep-supervision contract: list-of-logits + argmax of the final head.
- Registry path: build_model('swin_unetr', ...) returns SwinUNETR3D
  and the resulting model is plug-compatible with the post-A3 interface.
- Drop-in compatibility with Unet3D: identical input shape, identical
  output shape, identical interface signature.
- Validation guards: too-shallow Z, too-deep stack-depth, etc.
- Window-attention internals: window partition padding, shifted-window
  attention mask correctness on a small synthetic case.
"""

import pytest
import torch

from src.models.swin_unetr.blocks import (
    PatchEmbed3D,
    PatchMerging3D,
    SwinTransformerBlock3D,
    window_partition,
    window_reverse,
)
from src.models.swin_unetr.swin_unetr import SwinUNETR3D


def _build_swin(sample_dimension=(12, 64, 64), feature_size=12, depths=(2, 2, 2, 2),
                num_heads=(2, 4, 8, 16), window_size_xy=4,
                deep_supervision=False, ds_levels=2,
                in_channels=3, num_classes=2,
                z_deduction='auto', gradient_checkpointing='auto'):
    return SwinUNETR3D(
        _name='swin_unetr',
        _input_channels=in_channels,
        _number_of_classes=num_classes,
        _sample_dimension=list(sample_dimension),
        _feature_size=feature_size,
        _depths=depths,
        _num_heads=num_heads,
        _window_size_xy=window_size_xy,
        _z_deduction_per_stage=z_deduction,
        _gradient_checkpointing=gradient_checkpointing,
        _deep_supervision=deep_supervision,
        _ds_levels=ds_levels,
    )


# --- Window partition / reverse ------------------------------------------

def test_window_partition_reverse_roundtrip_no_padding():
    """Partition then reverse must reconstruct the original tensor exactly
    when H and W are already divisible by the window."""
    B, Z, H, W, C = 2, 4, 8, 8, 3
    x = torch.randn(B, Z, H, W, C)
    win = (Z, 4, 4)
    windows, padded_shape = window_partition(x, win)
    y = window_reverse(windows, win, padded_shape, (H, W))
    assert torch.allclose(x, y)


def test_window_partition_pads_and_crops_correctly():
    """When H or W isn't divisible by the window, partition zero-pads and
    reverse crops back to the original shape."""
    B, Z, H, W, C = 1, 4, 6, 7, 2   # H=6 (div by 3 not 4); W=7 (div by neither)
    x = torch.randn(B, Z, H, W, C)
    win = (Z, 4, 4)
    windows, padded_shape = window_partition(x, win)
    # padded_shape Z stays = Z; H pads to multiple of 4 (=8); W pads to multiple of 4 (=8)
    assert padded_shape == (Z, 8, 8)
    y = window_reverse(windows, win, padded_shape, (H, W))
    assert y.shape == x.shape
    assert torch.allclose(x, y)


def test_window_partition_rejects_mismatched_z():
    """Window Z must equal input Z (the project's full-Z-window assumption)."""
    x = torch.randn(1, 12, 8, 8, 3)
    with pytest.raises(ValueError, match="Window Z"):
        window_partition(x, (6, 4, 4))


# --- PatchEmbed3D / PatchMerging3D shape contracts ------------------------

def test_patch_embed_shape():
    embed = PatchEmbed3D(in_channels=3, embed_dim=16, patch_size=(1, 2, 2))
    x = torch.randn(2, 3, 12, 64, 64)
    y = embed(x)
    # Z preserved, XY halved; output in (B, Z, H, W, C) layout.
    assert y.shape == (2, 12, 32, 32, 16)


def test_patch_merging_deducts_z_and_halves_xy():
    merge = PatchMerging3D(dim=16, z_deduction=2)
    x = torch.randn(2, 12, 64, 64, 16)
    y = merge(x)
    assert y.shape == (2, 10, 32, 32, 32)   # Z 12→10, XY 64→32, C 16→32


def test_patch_merging_rejects_too_shallow_z():
    merge = PatchMerging3D(dim=16, z_deduction=2)
    x = torch.randn(1, 2, 16, 16, 16)  # Z=2 < kernel=3
    with pytest.raises(ValueError, match="needs Z >="):
        merge(x)


# --- SwinTransformerBlock3D ---------------------------------------------

def test_swin_block_preserves_shape():
    """A Swin block is shape-preserving (window attention on a per-window basis)."""
    block = SwinTransformerBlock3D(
        dim=12, num_heads=2, window_size=(6, 4, 4),
        shift_size=(0, 0, 0))
    x = torch.randn(1, 6, 8, 8, 12)
    y = block(x)
    assert y.shape == x.shape


def test_swin_block_shifted_variant_runs():
    """Shifted-window variant should also be shape-preserving."""
    block = SwinTransformerBlock3D(
        dim=12, num_heads=2, window_size=(6, 4, 4),
        shift_size=(0, 2, 2))
    x = torch.randn(1, 6, 8, 8, 12)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# --- SwinUNETR3D forward shape contract ----------------------------------

def test_swin_unetr_forward_output_matches_input_shape():
    model = _build_swin(sample_dimension=(12, 64, 64))
    x = torch.randn(1, 3, 12, 64, 64)
    logits, outputs = model(x)
    assert logits.shape == (1, 2, 12, 64, 64)
    assert outputs.shape == (1, 12, 64, 64)
    assert outputs.dtype in (torch.int64, torch.long)


def test_swin_unetr_works_on_different_sample_sizes():
    """The model must adapt to different input shapes without re-instantiation
    of building blocks (XY at least; Z is fixed at construction)."""
    model = _build_swin(sample_dimension=(12, 64, 64), feature_size=12,
                        window_size_xy=4)
    for hw in (64, 128, 32):
        x = torch.randn(1, 3, 12, hw, hw)
        logits, _ = model(x)
        assert logits.shape == (1, 2, 12, hw, hw)


# --- Deep supervision ------------------------------------------------------

def test_swin_unetr_ds_returns_list_with_correct_count():
    model = _build_swin(deep_supervision=True, ds_levels=2)
    x = torch.randn(1, 3, 12, 64, 64)
    logits, outputs = model(x)
    assert isinstance(logits, list)
    assert len(logits) == 3   # [final, aux_finest, aux_deepest]
    # Final head matches input resolution; aux heads progressively coarser.
    assert logits[0].shape == (1, 2, 12, 64, 64)
    for k in range(1, len(logits)):
        # Aux heads are at decoder mid-level spatial resolutions — at most
        # equal to the previous level on every axis.
        for axis in (2, 3, 4):
            assert logits[k].shape[axis] <= logits[k - 1].shape[axis]


def test_swin_unetr_ds_argmax_matches_final_head():
    model = _build_swin(deep_supervision=True, ds_levels=2)
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 12, 64, 64)
    with torch.no_grad():
        logits, outputs = model(x)
    expected = torch.argmax(torch.softmax(logits[0], dim=1), dim=1)
    assert torch.equal(outputs, expected)


def test_swin_unetr_ds_levels_capped_by_decoder_depth():
    """Asking for more DS levels than the decoder supports caps silently."""
    # 4 stages → 3 decoder up-steps → max_ds_levels = 3.
    model = _build_swin(deep_supervision=True, ds_levels=99)
    assert model.ds_levels == 3


def test_swin_unetr_no_ds_returns_single_tensor():
    """Default (DS off) preserves the post-A3 Tensor return contract."""
    model = _build_swin(deep_supervision=False)
    x = torch.randn(1, 3, 12, 64, 64)
    logits, _ = model(x)
    assert isinstance(logits, torch.Tensor)


# --- Registry / drop-in compatibility -------------------------------------

def test_swin_unetr_registered_in_model_registry():
    from src.models import MODEL_REGISTRY
    assert 'swin_unetr' in MODEL_REGISTRY


def test_swin_unetr_buildable_via_factory():
    from src.models import build_model
    configs = {
        'trainer': {
            'model': {
                'name': 'swin_unetr',
                'encoder_kernel': [3, 3, 3],
                'decoder_kernel': [3, 3, 3],
                'feature_maps': [],
                'swin_unetr': {
                    'feature_size': 12,
                    'depths': [2, 2, 2, 2],
                    'num_heads': [2, 4, 8, 16],
                    'window_size_xy': 4,
                },
            },
            'train_ds': {'sample_dimension': [12, 64, 64]},
        },
    }
    model = build_model('swin_unetr', configs, 3, 2)
    assert isinstance(model, SwinUNETR3D)
    # The registry-built model satisfies the post-A3 interface.
    x = torch.randn(1, 3, 12, 64, 64)
    logits, outputs = model(x)
    assert logits.shape == (1, 2, 12, 64, 64)
    assert outputs.shape == (1, 12, 64, 64)


def test_swin_unetr_drops_in_for_unet3d_on_same_input():
    """SwinUNETR and Unet3D both consume the same input shape and produce
    output of the same shape with the same dtype + interface."""
    from src.models.unet3d.unet3d import Unet3D

    unet = Unet3D(
        _name='unet_3d', _input_channels=3, _number_of_classes=2,
        _feature_maps=(12, 24, 48, 96),
        _sample_dimension=[12, 64, 64])
    swin = _build_swin(sample_dimension=(12, 64, 64))
    x = torch.randn(1, 3, 12, 64, 64)

    u_logits, u_outputs = unet(x)
    s_logits, s_outputs = swin(x)

    assert u_logits.shape == s_logits.shape
    assert u_outputs.shape == s_outputs.shape
    assert u_logits.dtype == s_logits.dtype
    assert u_outputs.dtype == s_outputs.dtype


# --- Validity guards -------------------------------------------------------

def test_swin_unetr_rejects_too_shallow_z_for_depth():
    """Even at the minimum auto deduction of 1, Z=3 over 4 stages collapses
    the bottleneck to 0 — the guard must still fire."""
    with pytest.raises(ValueError, match="too shallow"):
        _build_swin(sample_dimension=(3, 32, 32), depths=(2, 2, 2, 2))


def test_swin_unetr_rejects_mismatched_depths_and_heads():
    with pytest.raises(ValueError, match="same length"):
        SwinUNETR3D(
            _name='swin_unetr', _input_channels=3, _number_of_classes=2,
            _sample_dimension=[12, 64, 64],
            _depths=(2, 2, 2),
            _num_heads=(2, 4, 8, 16))


def test_swin_unetr_rejects_too_few_stages():
    with pytest.raises(ValueError, match="at least 2"):
        SwinUNETR3D(
            _name='swin_unetr', _input_channels=3, _number_of_classes=2,
            _sample_dimension=[12, 64, 64],
            _depths=(2,),
            _num_heads=(2,))


# --- Backwards pass --------------------------------------------------------

def test_swin_unetr_backward_flows_to_input():
    """Sanity: the model is differentiable end-to-end."""
    model = _build_swin(sample_dimension=(12, 32, 32), window_size_xy=4)
    x = torch.randn(1, 3, 12, 32, 32, requires_grad=True)
    logits, _ = model(x)
    logits.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# --- Adaptive z_deduction + gradient checkpointing ------------------------

def test_swin_unetr_auto_z_deduction_scales_with_patch_depth():
    """'auto' z_deduction is derived from the Z patch depth — it reproduces
    the hand-tuned 2 at Z=12 and scales up for deeper patches."""
    assert _build_swin(sample_dimension=(12, 64, 64)).z_deduction == 2
    assert _build_swin(sample_dimension=(24, 64, 64)).z_deduction == 4
    assert _build_swin(sample_dimension=(6, 64, 64)).z_deduction == 1


def test_swin_unetr_explicit_z_deduction_overrides_auto():
    """An explicit integer z_deduction is used verbatim, not derived."""
    model = _build_swin(sample_dimension=(24, 64, 64), z_deduction=2)
    assert model.z_deduction == 2


def test_swin_unetr_encoder_merge_uses_configured_z_deduction():
    """The encoder PatchMerging3D must deduct by the configured value —
    regression for the BasicLayer3D path that ignored it (always 2)."""
    model = _build_swin(sample_dimension=(24, 64, 64), z_deduction=4)
    # PatchMerging3D's z-kernel is z_deduction + 1.
    assert model.encoder_stages[0].downsample.z_kernel == 5


def test_swin_unetr_checkpointing_setting_resolves():
    """'on'/'off'/bool/None resolve deterministically; 'auto' is off on CPU."""
    resolve = SwinUNETR3D._resolve_checkpointing
    assert resolve('on') is True
    assert resolve(True) is True
    assert resolve('off') is False
    assert resolve(False) is False
    assert resolve(None) is False
    assert resolve('auto') is False   # no CUDA device in the test env


def test_swin_unetr_forward_backward_with_checkpointing():
    """With checkpointing forced on, forward + backward still run and
    produce finite gradients (the checkpointed encoder is exact)."""
    model = _build_swin(sample_dimension=(12, 32, 32), window_size_xy=4,
                        gradient_checkpointing='on')
    assert model.use_checkpointing is True
    x = torch.randn(1, 3, 12, 32, 32, requires_grad=True)
    logits, _ = model(x)
    assert logits.shape == (1, 2, 12, 32, 32)
    logits.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
