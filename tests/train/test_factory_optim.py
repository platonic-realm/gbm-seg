"""Optimiser + scheduler factory regression. Locks in §C1.1.

The factory now dispatches on ``configs.trainer.optim.name`` (adam / sgd)
and ``configs.trainer.scheduler.name`` (reduce_on_plateau / poly_decay).
Tests exercise both paths plus default-fallback behaviour and the
unknown-name error path.
"""

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR, ReduceLROnPlateau


def _factory_with(trainer_overrides: dict):
    """Build a Factory from a minimal configs dict + the given trainer overrides."""
    from src.train.factory import Factory

    base_configs = {
        'root_path': '/tmp',
        'trainer': {
            'result_path': 'results-train/',
            'snapshot_path': 'snapshots/',
            'model': {
                'name': 'unet_3d',
                'encoder_kernel': [3, 3, 3],
                'decoder_kernel': [3, 3, 3],
                'feature_maps': [4, 8],
            },
            'epochs': 10,
            'optim': {'name': 'adam', 'lr': 1e-4},
            'train_ds': {'sample_dimension': [4, 16, 16]},
            'dp': False,
            'device': 'cpu',
        },
    }
    # Recursive update for the relevant sub-dicts only.
    for key, value in trainer_overrides.items():
        if isinstance(value, dict):
            base_configs['trainer'].setdefault(key, {})
            base_configs['trainer'][key].update(value)
        else:
            base_configs['trainer'][key] = value
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


def test_scheduler_default_is_reduce_on_plateau():
    """If `trainer.scheduler` is missing entirely, default to ReduceLROnPlateau
    with the prior behaviour (mode=max). Backwards-compat for old experiments."""
    factory = _factory_with({'optim': {'name': 'adam', 'lr': 1e-4}})
    opt = factory.createOptimizer(_TinyNet(), None)
    sched = factory.createScheduler(opt)
    assert isinstance(sched, ReduceLROnPlateau)
    assert sched.mode == 'max'


def test_scheduler_reduce_on_plateau_explicit_fields():
    factory = _factory_with({
        'optim': {'name': 'adam', 'lr': 1e-4},
        'scheduler': {'name': 'reduce_on_plateau', 'mode': 'min',
                      'factor': 0.5, 'patience': 5},
    })
    opt = factory.createOptimizer(_TinyNet(), None)
    sched = factory.createScheduler(opt)
    assert isinstance(sched, ReduceLROnPlateau)
    assert sched.mode == 'min'
    assert sched.factor == 0.5
    assert sched.patience == 5


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
