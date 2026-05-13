"""Continuity-loss axis regression.

Locks in §1.5. The pre-fix code took softmax over W (``dim=-1``) and then
sliced along the *class* axis instead of depth — so it penalised the wrong
thing entirely. After the fix, the loss must be zero when adjacent depth
slices are identical, zero when variation is purely along W, and positive
when variation is along D.
"""

import torch

from src.train.losses.loss_cont import continuity_loss_diff


def test_identical_depth_slices_zero():
    x = torch.zeros(1, 2, 4, 8, 8)
    x[:, 0, :, :, :] = 5.0  # constant along D
    assert continuity_loss_diff(x).item() == 0.0


def test_variation_only_along_W_zero():
    """Softmax must be over the class dim; otherwise W-variation leaks in."""
    x = torch.zeros(1, 2, 4, 8, 8)
    x[:, 0, :, :, :4] = 10.0
    x[:, 1, :, :, 4:] = 10.0  # different along W, identical along D
    assert continuity_loss_diff(x).item() == 0.0


def test_variation_along_D_positive():
    x = torch.zeros(1, 2, 4, 8, 8)
    x[:, 0, ::2, :, :] = 10.0
    x[:, 1, 1::2, :, :] = 10.0
    assert continuity_loss_diff(x).item() > 0.5


def test_rejects_wrong_rank():
    with __import__('pytest').raises(AssertionError):
        continuity_loss_diff(torch.zeros(1, 2, 4, 8))  # 4D, not 5D


def test_cont_loss_module_runs_and_backprops():
    """End-to-end ContLoss: forward pass produces a finite scalar and the
    learned alpha/beta weights actually receive gradients."""
    from torch import nn

    from src.train.losses.loss_cont import ContLoss

    weights = torch.tensor([1.0, 1.0])
    loss_fn = ContLoss(nn.CrossEntropyLoss(weight=weights))

    logits = torch.randn(2, 2, 4, 8, 8, requires_grad=True)
    labels = torch.randint(0, 2, (2, 4, 8, 8))

    out = loss_fn(logits, labels)
    assert torch.isfinite(out).item()
    out.backward()
    assert loss_fn.w_alpha.grad is not None
    assert loss_fn.w_beta.grad is not None


def test_cont_loss_rejects_bad_minimums():
    import pytest
    from torch import nn

    with pytest.raises(ValueError):
        from src.train.losses.loss_cont import ContLoss
        ContLoss(nn.CrossEntropyLoss(), min_alpha=0.7, min_beta=0.5)
