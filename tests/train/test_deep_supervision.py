"""Deep supervision regression. Locks in §C1.2.

Three layers of behaviour to pin:
1. Unet3D with deep_supervision=False returns a single logits tensor (backward-compat).
2. Unet3D with deep_supervision=True returns a list [final, aux_1, ..., aux_L] at
   geometrically-decreasing spatial resolutions, and the final head matches the non-DS forward.
3. DeepSupervisionLoss combines per-level losses against correctly downsampled labels,
   degrades to the base loss when the model returns a single tensor, and is end-to-end
   differentiable.
"""

import pytest
import torch
from torch import nn

from src.models.unet3d.unet3d import Unet3D
from src.train.losses.loss_ds import DeepSupervisionLoss, default_ds_weights


def _build_unet(ds: bool, ds_levels: int = 2,
                feature_maps=(8, 16, 32, 64), sample_dim=(12, 16, 16)):
    # Z=12 supports 4-stage encoder with default deduct=2 (12 → 10 → 8 → 6).
    # 2-stage variants for the capped-DS test use Z=12 too (just don't deduct
    # past zero).
    return Unet3D(
        _name='unet_3d',
        _input_channels=3,
        _number_of_classes=2,
        _feature_maps=feature_maps,
        _sample_dimension=list(sample_dim),
        _deep_supervision=ds,
        _ds_levels=ds_levels,
    )


# --- Default DS weights ----------------------------------------------------

def test_default_ds_weights_sum_to_one():
    for n in (1, 2, 3, 5):
        ws = default_ds_weights(n)
        assert abs(sum(ws) - 1.0) < 1e-6


def test_default_ds_weights_geometric_decay():
    ws = default_ds_weights(3)
    # Ratios between consecutive weights are 1/2.
    assert abs(ws[0] / ws[1] - 2.0) < 1e-6
    assert abs(ws[1] / ws[2] - 2.0) < 1e-6


# --- Unet3D shapes ---------------------------------------------------------

def test_unet3d_no_ds_returns_single_tensor():
    model = _build_unet(ds=False)
    x = torch.randn(1, 3, 12, 16, 16)
    logits, outputs = model(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 2, 12, 16, 16)
    assert outputs.shape == (1, 12, 16, 16)


def test_unet3d_with_ds_returns_list():
    model = _build_unet(ds=True, ds_levels=2)
    x = torch.randn(1, 3, 12, 16, 16)
    logits, outputs = model(x)
    assert isinstance(logits, list)
    assert len(logits) == 3, "expected [final, aux_0, aux_1]"


def test_unet3d_with_ds_aux_logits_have_decreasing_spatial_resolution():
    model = _build_unet(ds=True, ds_levels=2)
    x = torch.randn(1, 3, 12, 16, 16)
    logits, _ = model(x)
    # Final head at full resolution
    assert logits[0].shape == (1, 2, 12, 16, 16)
    # Aux heads at progressively smaller XY (Z preserved by (1,2,2) pooling).
    # Decoder 0 output: X/4, Y/4 = 4, 4. Decoder 1 output: X/2, Y/2 = 8, 8.
    for k in range(1, len(logits)):
        prev_xy = logits[k - 1].shape[-2:]
        cur_xy = logits[k].shape[-2:]
        assert cur_xy[0] <= prev_xy[0]
        assert cur_xy[1] <= prev_xy[1]
        # And the aux heads keep the class dim.
        assert logits[k].shape[1] == 2


def test_unet3d_with_ds_argmax_matches_final_head():
    """The returned ``outputs`` is the argmax of the FINAL head, regardless of DS."""
    torch.manual_seed(0)
    model = _build_unet(ds=True, ds_levels=2)
    model.eval()
    x = torch.randn(1, 3, 12, 16, 16)
    with torch.no_grad():
        logits, outputs = model(x)
    expected = torch.argmax(torch.softmax(logits[0], dim=1), dim=1)
    assert torch.equal(outputs, expected)


def test_unet3d_ds_levels_capped_by_decoder_depth():
    """Asking for more DS levels than the decoder can support is silently capped."""
    # 3 feature_maps → 2 decoder layers → max_ds_levels = 1.
    # sample_dim=(12, 16, 16) is deep enough for 3-stage encoder with deduct=2
    # (Z trajectory: 12 → 10 → 8).
    model = _build_unet(ds=True, ds_levels=99,
                        feature_maps=(8, 16, 32), sample_dim=(12, 16, 16))
    assert model.ds_levels == 1


# --- DeepSupervisionLoss ---------------------------------------------------

def _ce():
    return nn.CrossEntropyLoss()


def test_ds_loss_degrades_to_base_when_single_tensor():
    base = _ce()
    wrapped = DeepSupervisionLoss(base)
    logits = torch.randn(2, 2, 4, 8, 8)
    labels = torch.randint(0, 2, (2, 4, 8, 8))
    assert torch.isclose(wrapped(logits, labels), base(logits, labels), atol=1e-6)


def test_ds_loss_combines_per_level_against_downsampled_labels():
    base = _ce()
    wrapped = DeepSupervisionLoss(base)
    # Two logits tensors, both shape match labels at different resolutions.
    labels = torch.randint(0, 2, (2, 4, 16, 16))
    logits_final = torch.randn(2, 2, 4, 16, 16)
    logits_aux = torch.randn(2, 2, 4, 8, 8)

    loss_combined = wrapped([logits_final, logits_aux], labels)

    # Manually compute expected: w0 * CE(final, labels) + w1 * CE(aux, labels_half)
    import torch.nn.functional as F
    labels_half = F.interpolate(labels.unsqueeze(1).float(),
                                size=(4, 8, 8), mode='nearest').squeeze(1).long()
    w0, w1 = default_ds_weights(2)
    expected = w0 * base(logits_final, labels) + w1 * base(logits_aux, labels_half)
    assert torch.isclose(loss_combined, expected, atol=1e-6)


def test_ds_loss_rejects_weight_count_mismatch():
    wrapped = DeepSupervisionLoss(_ce(), weights=[1.0, 0.5])
    labels = torch.randint(0, 2, (2, 4, 8, 8))
    logits = [torch.randn(2, 2, 4, 8, 8)] * 3  # 3 heads but only 2 weights
    with pytest.raises(ValueError, match="DS weight count"):
        wrapped(logits, labels)


def test_ds_loss_backward_flows_to_all_inputs():
    wrapped = DeepSupervisionLoss(_ce())
    labels = torch.randint(0, 2, (2, 4, 16, 16))
    logits_final = torch.randn(2, 2, 4, 16, 16, requires_grad=True)
    logits_aux = torch.randn(2, 2, 4, 8, 8, requires_grad=True)
    out = wrapped([logits_final, logits_aux], labels)
    out.backward()
    assert logits_final.grad is not None and torch.isfinite(logits_final.grad).all()
    assert logits_aux.grad is not None and torch.isfinite(logits_aux.grad).all()
