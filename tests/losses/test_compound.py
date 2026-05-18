"""CompoundLoss regression. Enables the catalog's literal A2 (b)–(d) configs
(Dice + CrossEntropy, Dice + ContLoss, etc.) without bespoke classes."""

import pytest
import torch
from torch import nn

from src.train.losses.loss_compound import CompoundLoss
from src.train.losses.loss_dice import DiceLoss


def _logits_and_labels():
    logits = torch.randn(2, 2, 4, 8, 8)
    labels = torch.randint(0, 2, (2, 4, 8, 8))
    return logits, labels


def test_compound_loss_returns_weighted_sum_of_components():
    weights = torch.tensor([1.0, 1.0])
    dice = DiceLoss(_weights=weights)
    ce = nn.CrossEntropyLoss(weight=weights)

    compound = CompoundLoss([(dice, 1.0), (ce, 0.5)])
    logits, labels = _logits_and_labels()
    out = compound(logits, labels)
    expected = dice(logits, labels) + 0.5 * ce(logits, labels)
    assert torch.isclose(out, expected, atol=1e-6)


def test_compound_loss_single_component_equals_base_loss():
    weights = torch.tensor([1.0, 1.0])
    base = nn.CrossEntropyLoss(weight=weights)
    compound = CompoundLoss([(base, 1.0)])
    logits, labels = _logits_and_labels()
    assert torch.isclose(compound(logits, labels), base(logits, labels), atol=1e-6)


def test_compound_loss_zero_weight_drops_term():
    weights = torch.tensor([1.0, 1.0])
    dice = DiceLoss(_weights=weights)
    ce = nn.CrossEntropyLoss(weight=weights)

    compound = CompoundLoss([(dice, 1.0), (ce, 0.0)])
    logits, labels = _logits_and_labels()
    out = compound(logits, labels)
    expected = dice(logits, labels)
    assert torch.isclose(out, expected, atol=1e-6)


def test_compound_loss_empty_components_rejected():
    with pytest.raises(ValueError, match="at least one component"):
        CompoundLoss([])


def test_compound_loss_backward_flows_through_all_components():
    weights = torch.tensor([1.0, 1.0])
    dice = DiceLoss(_weights=weights)
    ce = nn.CrossEntropyLoss(weight=weights)

    compound = CompoundLoss([(dice, 1.0), (ce, 1.0)])
    logits = torch.randn(2, 2, 4, 8, 8, requires_grad=True)
    labels = torch.randint(0, 2, (2, 4, 8, 8))
    compound(logits, labels).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_compound_loss_to_device_moves_sub_losses():
    """sub_losses lives on nn.ModuleList so .to() propagates."""
    weights = torch.tensor([1.0, 1.0])
    dice = DiceLoss(_weights=weights)
    ce = nn.CrossEntropyLoss(weight=weights)
    compound = CompoundLoss([(dice, 1.0), (ce, 1.0)])
    compound.to('cpu')  # smoke test — should not raise
    assert compound.sub_losses[0] is dice
    assert compound.sub_losses[1] is ce


def test_factory_compound_loss_builds_from_config(monkeypatch, tmp_path):
    """End-to-end: factory.createLoss with loss=Compound returns CompoundLoss."""
    from src.train.factory import Factory

    configs = {
        'root_path': str(tmp_path),
        'trainer': {
            'logging': {'result_path': 'results-train/'},
            'runtime': {'device': 'cpu'},
            'optimization': {
                'loss': {
                    'name': 'Compound',
                    'weights': [1.0, 1.0],
                    'compound': [
                        {'name': 'Dice', 'weight': 1.0},
                        {'name': 'CrossEntropy', 'weight': 1.0},
                    ],
                },
            },
        },
    }
    factory = Factory(configs)
    loss = factory.createLoss()
    assert isinstance(loss, CompoundLoss)
    assert len(loss.sub_losses) == 2
    assert loss.weights == [1.0, 1.0]


def test_factory_compound_loss_with_cont_sub_loss(tmp_path):
    """A2 config (c): Dice + ContLoss (CE+continuity), achievable via Compound."""
    from src.train.factory import Factory
    from src.train.losses.loss_cont import ContLoss

    configs = {
        'root_path': str(tmp_path),
        'trainer': {
            'logging': {'result_path': 'results-train/'},
            'runtime': {'device': 'cpu'},
            'optimization': {
                'loss': {
                    'name': 'Compound',
                    'weights': [1.0, 1.0],
                    'compound': [
                        {'name': 'Dice', 'weight': 1.0},
                        {'name': 'Cont', 'weight': 1.0,
                         'params': {'cont_alpha': 0.7, 'cont_beta': 0.3}},
                    ],
                },
            },
        },
    }
    factory = Factory(configs)
    loss = factory.createLoss()
    assert isinstance(loss, CompoundLoss)
    assert isinstance(loss.sub_losses[1], ContLoss)
    assert loss.sub_losses[1].alpha == 0.7
    assert loss.sub_losses[1].beta == 0.3
