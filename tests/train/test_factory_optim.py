"""Optimiser + scheduler factory regression.

The factory dispatches on ``configs.trainer.optim.name`` (adam / sgd)
and ``configs.trainer.scheduler.name`` (``poly_decay`` / ``none``).
Tests exercise both paths plus default-fallback behaviour and the
unknown-name error path.
"""

import math
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR


def _factory_with(optimization_overrides: dict):
    """Build a Factory from a minimal configs dict + the given
    ``trainer.optimization`` overrides (``optim``, ``scheduler``, ``epochs``)."""
    from src.train.factory import Factory

    base_configs = {
        'root_path': '/tmp',
        'trainer': {
            'logging': {
                'result_path': 'results-train/',
                'snapshot_path': 'snapshots/',
            },
            'model': {
                'name': 'unet_3d',
                'unet_3d': {
                    'encoder_kernel': [3, 3, 3],
                    'decoder_kernel': [3, 3, 3],
                    'feature_maps': [4, 8],
                },
            },
            'optimization': {
                'epochs': 10,
                'optim': {'name': 'adam', 'lr': 1e-4},
            },
            'data': {'train_ds': {'sample_dimension': [4, 16, 16]}},
            'runtime': {'dp': False, 'device': 'cpu'},
        },
    }
    # Recursive update for the relevant sub-dicts only.
    optimization = base_configs['trainer']['optimization']
    for key, value in optimization_overrides.items():
        if isinstance(value, dict):
            optimization.setdefault(key, {})
            optimization[key].update(value)
        else:
            optimization[key] = value
    return Factory(base_configs)


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_optim_adam_returns_adam():
    factory = _factory_with({'optim': {'name': 'adam', 'lr': 1e-4}})
    opt = factory.createOptimizer(_TinyNet(), None)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]['lr'] == 1e-4


def test_optim_sgd_returns_sgd_with_momentum():
    factory = _factory_with({'optim': {'name': 'sgd', 'lr': 1e-2}})
    opt = factory.createOptimizer(_TinyNet(), None)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]['lr'] == 1e-2
    # Default momentum is nnU-Net's 0.99 when not specified.
    assert opt.param_groups[0]['momentum'] == 0.99


def test_optim_sgd_respects_explicit_momentum_and_wd():
    factory = _factory_with({'optim': {
        'name': 'sgd', 'lr': 1e-2, 'momentum': 0.95, 'weight_decay': 5e-4,
        'nesterov': True}})
    opt = factory.createOptimizer(_TinyNet(), None)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]['momentum'] == 0.95
    assert opt.param_groups[0]['weight_decay'] == 5e-4
    assert opt.param_groups[0]['nesterov'] is True


def test_optim_unknown_name_raises():
    factory = _factory_with({'optim': {'name': 'rmsprop', 'lr': 1e-4}})
    with pytest.raises(NotImplementedError, match="Unknown optimiser"):
        factory.createOptimizer(_TinyNet(), None)


def test_scheduler_default_is_poly_decay():
    """If `trainer.scheduler.name` is missing, default to poly_decay
    with total_iters = trainer.optimization.epochs."""
    factory = _factory_with({
        'optim': {'name': 'adam', 'lr': 1e-4},
        'scheduler': {},
    })
    opt = factory.createOptimizer(_TinyNet(), None)
    sched = factory.createScheduler(opt)
    assert isinstance(sched, PolynomialLR)
    assert sched.total_iters == 10  # epochs in the minimal config


def test_scheduler_none_returns_none():
    """`scheduler: null` (parsed to None) disables scheduling."""
    factory = _factory_with({
        'optim': {'name': 'adam', 'lr': 1e-4},
        'scheduler': None,
    })
    opt = factory.createOptimizer(_TinyNet(), None)
    assert factory.createScheduler(opt) is None


def test_scheduler_named_none_returns_none():
    """``scheduler: {name: none}`` also disables scheduling."""
    factory = _factory_with({
        'optim': {'name': 'adam', 'lr': 1e-4},
        'scheduler': {'name': 'none'},
    })
    opt = factory.createOptimizer(_TinyNet(), None)
    assert factory.createScheduler(opt) is None


def test_scheduler_poly_decay_returns_polynomial_lr():
    factory = _factory_with({
        'optim': {'name': 'sgd', 'lr': 1e-2},
        'scheduler': {'name': 'poly_decay', 'total_iters': 100, 'power': 0.9},
    })
    opt = factory.createOptimizer(_TinyNet(), None)
    sched = factory.createScheduler(opt)
    assert isinstance(sched, PolynomialLR)
    assert sched.total_iters == 100
    assert sched.power == 0.9


def test_scheduler_poly_decay_defaults_total_iters_to_epochs():
    factory = _factory_with({
        'optim': {'name': 'sgd', 'lr': 1e-2},
        'scheduler': {'name': 'poly_decay'},
    })
    opt = factory.createOptimizer(_TinyNet(), None)
    sched = factory.createScheduler(opt)
    # `epochs` was 10 in the minimal config.
    assert sched.total_iters == 10
    assert sched.power == 0.9  # nnU-Net default


def test_scheduler_unknown_name_raises():
    factory = _factory_with({
        'optim': {'name': 'adam', 'lr': 1e-4},
        'scheduler': {'name': 'cosine_annealing'},
    })
    opt = factory.createOptimizer(_TinyNet(), None)
    with pytest.raises(NotImplementedError, match="Unknown scheduler"):
        factory.createScheduler(opt)


def test_poly_decay_lr_actually_decays():
    """End-to-end: stepping the PolynomialLR scheduler reduces the LR per
    nnU-Net's (1 - t/T)^power. After half the iterations, LR should be
    roughly 0.5^0.9 of the initial value."""
    factory = _factory_with({
        'optim': {'name': 'sgd', 'lr': 0.01},
        'scheduler': {'name': 'poly_decay', 'total_iters': 10, 'power': 0.9},
    })
    net = _TinyNet()
    opt = factory.createOptimizer(net, None)
    sched = factory.createScheduler(opt)

    # Take one optimiser step first so PyTorch doesn't warn about
    # "scheduler.step() called before optimizer.step()". We don't need
    # a real forward pass — any loss against the parameters works.
    next(iter(net.parameters())).sum().backward()
    opt.step()

    initial_lr = opt.param_groups[0]['lr']
    for _ in range(5):
        sched.step()
    half_lr = opt.param_groups[0]['lr']

    assert half_lr < initial_lr
    # Approximate (1 - 5/10)^0.9 ≈ 0.536
    assert abs(half_lr / initial_lr - 0.5**0.9) < 0.05


# --- DDP effective-batch LR scaling --------------------------------------


def test_optim_adam_lr_sqrt_scaled_under_ddp():
    """Under DDP the effective batch is per-rank batch x world_size, so
    Adam's LR is scaled by sqrt(world_size) — the square-root rule that
    holds for adaptive optimisers (linear scaling is SGD-only)."""
    factory = _factory_with({'optim': {'name': 'adam', 'lr': 1e-4}})
    with patch('src.train.factory.get_world_size', return_value=8):
        opt = factory.createOptimizer(_TinyNet(), None)
    assert opt.param_groups[0]['lr'] == pytest.approx(1e-4 * math.sqrt(8))


def test_optim_sgd_lr_linear_scaled_under_ddp():
    """Under DDP, SGD's LR is scaled linearly by world_size — the linear
    scaling rule (Goyal et al.), which holds for SGD but not Adam."""
    factory = _factory_with({'optim': {'name': 'sgd', 'lr': 1e-2}})
    with patch('src.train.factory.get_world_size', return_value=4):
        opt = factory.createOptimizer(_TinyNet(), None)
    assert opt.param_groups[0]['lr'] == pytest.approx(1e-2 * 4)


def test_optim_lr_unscaled_in_single_process():
    """world_size == 1 (no DDP) → the config LR is used verbatim."""
    factory = _factory_with({'optim': {'name': 'adam', 'lr': 1e-4}})
    with patch('src.train.factory.get_world_size', return_value=1):
        opt = factory.createOptimizer(_TinyNet(), None)
    assert opt.param_groups[0]['lr'] == pytest.approx(1e-4)
