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
    """End-to-end ContLoss: forward produces a finite scalar; gradient
    flows back to the input logits. After the A2-precursor refactor,
    ContLoss no longer has learnable alpha/beta — its weights are
    fixed hyperparameters, so we just verify the loss is differentiable
    w.r.t. its inputs."""
    from torch import nn

    from src.train.losses.loss_cont import ContLoss

    weights = torch.tensor([1.0, 1.0])
    loss_fn = ContLoss(nn.CrossEntropyLoss(weight=weights),
                       _alpha=0.7, _beta=0.3)

    logits = torch.randn(2, 2, 4, 8, 8, requires_grad=True)
    labels = torch.randint(0, 2, (2, 4, 8, 8))

    out = loss_fn(logits, labels)
    assert torch.isfinite(out).item()
    out.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_cont_loss_has_no_learnable_params():
    """A2-precursor: confirm ContLoss has no nn.Parameters anymore.
    The factory's `list(_model.parameters()) + list(_loss.parameters())`
    pattern must therefore add zero parameters from the loss."""
    from torch import nn

    from src.train.losses.loss_cont import ContLoss

    loss_fn = ContLoss(nn.CrossEntropyLoss())
    assert list(loss_fn.parameters()) == []
    # The fixed weights live on the module as plain floats:
    assert isinstance(loss_fn.alpha, float)
    assert isinstance(loss_fn.beta, float)


def test_cont_loss_respects_alpha_beta_extremes():
    """At (alpha=1, beta=0) the loss is exactly CE.
    At (alpha=0, beta=1) the loss is exactly the continuity term."""
    from torch import nn

    from src.train.losses.loss_cont import ContLoss, continuity_loss_diff

    weights = torch.tensor([1.0, 1.0])
    ce = nn.CrossEntropyLoss(weight=weights)

    logits = torch.randn(2, 2, 4, 8, 8)
    labels = torch.randint(0, 2, (2, 4, 8, 8))

    only_ce = ContLoss(ce, _alpha=1.0, _beta=0.0)(logits, labels)
    only_ct = ContLoss(ce, _alpha=0.0, _beta=1.0)(logits, labels)

    expected_ce = ce(logits, labels)
    expected_ct = continuity_loss_diff(logits)

    assert torch.isclose(only_ce, expected_ce, atol=1e-6)
    assert torch.isclose(only_ct, expected_ct, atol=1e-6)
