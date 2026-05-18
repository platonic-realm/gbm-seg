"""Model-registry regression. Locks in §B1.2.

The factory dispatches through ``src.models.MODEL_REGISTRY`` rather than
hard-instantiating Unet3D. Tests cover: the registry has the expected
default entry, unknown names raise NotImplementedError, the build callable
returns a working Unet3D, and a *third-party* model registered at runtime
is successfully constructed by the factory (proves the extension point
works for the future custom SwinUNETR / variant additions).
"""

import pytest
import torch
from torch import nn

from src.models import MODEL_REGISTRY, build_model
from src.models.unet3d.unet3d import Unet3D


def _minimal_configs():
    """A configs dict with the minimum keys needed by the unet_3d build().

    Uses sample_dim=[12, 16, 16] so the Z-deduct pooling has enough headroom
    (Z trajectory: 12 → 10 → 8 for 3-stage encoder). The 2-stage Z=4 case is
    degenerate — the output Z stops at the bottleneck Z because the single
    decoder layer doesn't upsample.
    """
    return {
        'trainer': {
            'model': {
                'name': 'unet_3d',
                'unet_3d': {
                    'encoder_kernel': [3, 3, 3],
                    'decoder_kernel': [3, 3, 3],
                    'feature_maps': [4, 8, 16],
                },
            },
            'optimization': {},
            'data': {
                'train_ds': {
                    'sample_dimension': [12, 16, 16],
                },
            },
        },
    }


def test_unet3d_registered_by_default():
    assert 'unet_3d' in MODEL_REGISTRY


def test_build_model_unknown_name_raises():
    with pytest.raises(NotImplementedError, match="Unknown model"):
        build_model('nonexistent', _minimal_configs(), 3, 2)


def test_build_model_returns_unet3d():
    model = build_model('unet_3d', _minimal_configs(), 3, 2)
    assert isinstance(model, Unet3D)


def test_build_model_forward_pass_works():
    """Sanity: the registry-built model can do a forward pass and matches
    the post-A3 interface (returns logits, outputs)."""
    model = build_model('unet_3d', _minimal_configs(), 3, 2)
    x = torch.randn(1, 3, 12, 16, 16)
    logits, outputs = model(x)
    assert logits.shape == (1, 2, 12, 16, 16)
    assert outputs.shape == (1, 12, 16, 16)


def test_registry_extension_point(monkeypatch):
    """A new model can be added at runtime by registering its build callable.
    Validates the future-extension path (custom SwinUNETR etc.)."""

    class DummyModel(nn.Module):
        def __init__(self, in_ch, num_classes):
            super().__init__()
            self.in_ch = in_ch
            self.num_classes = num_classes
            self.last_layer = nn.Conv3d(in_ch, num_classes, kernel_size=1)

        def forward(self, x):
            logits = self.last_layer(x)
            return logits, logits.argmax(dim=1)

    def build_dummy(configs, in_ch, num_classes):
        return DummyModel(in_ch, num_classes)

    monkeypatch.setitem(MODEL_REGISTRY, 'dummy', build_dummy)
    assert 'dummy' in MODEL_REGISTRY

    model = build_model('dummy', _minimal_configs(), 3, 2)
    assert isinstance(model, DummyModel)
    assert model.in_ch == 3
    assert model.num_classes == 2

    x = torch.randn(1, 3, 4, 8, 8)
    logits, outputs = model(x)
    assert logits.shape == (1, 2, 4, 8, 8)
    assert outputs.shape == (1, 4, 8, 8)
