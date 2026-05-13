"""W&B integration regression. Locks in §E2.

wandb itself is not installed in CI (or in this venv), so these tests
inject a stub ``wandb`` module into ``sys.modules`` and verify the
intended call chain: ``MetricWandb.log`` translates a metric dict into
namespaced ``wandb.log`` payloads, the factory plumbs through to a
``MetricWandb`` instance only when the config flag is on, and the
``Snapper.save`` artifact-upload path is a no-op when no run is active.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

# --- Stub wandb -----------------------------------------------------------

@pytest.fixture
def fake_wandb(monkeypatch):
    """Inject a fake `wandb` module into sys.modules.

    Provides ``wandb.log``, ``wandb.init``, ``wandb.save``, ``wandb.finish``
    as MagicMocks and an ``run`` attribute that defaults to None so the
    "no active run" code paths are exercised by default.
    """
    fake = types.ModuleType('wandb')
    fake.log = MagicMock()
    fake.init = MagicMock()
    fake.save = MagicMock()
    fake.finish = MagicMock()
    fake.run = None  # callers set this to a truthy mock when a run is "active"
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    yield fake


# --- MetricWandb ----------------------------------------------------------

def test_metric_wandb_log_namespaces_metrics_by_tag(fake_wandb):
    fake_wandb.run = MagicMock()  # active run

    from src.utils.metrics.log.metric_wandb import MetricWandb
    backend = MetricWandb()
    backend.log(_epoch=3, _step=42, _seen_labels=336,
                _tag='train', _metrics={'Loss': 0.5, 'Dice': 0.8})

    fake_wandb.log.assert_called_once()
    payload, kwargs = fake_wandb.log.call_args
    assert payload[0] == {
        'train/Loss': 0.5,
        'train/Dice': 0.8,
        'epoch': 3,
        'seen_labels': 336,
    }
    assert kwargs == {'step': 42}


def test_metric_wandb_log_warns_with_no_active_run(fake_wandb, caplog):
    fake_wandb.run = None  # explicit no-run

    from src.utils.metrics.log.metric_wandb import MetricWandb
    backend = MetricWandb()
    backend.log(_epoch=0, _step=1, _seen_labels=8,
                _tag='train', _metrics={'Loss': 0.5})

    assert not fake_wandb.log.called
    assert "no active wandb run" in caplog.text.lower()


def test_metric_wandb_log_coerces_torch_tensors(fake_wandb):
    fake_wandb.run = MagicMock()

    from src.utils.metrics.log.metric_wandb import MetricWandb
    backend = MetricWandb()
    backend.log(_epoch=0, _step=1, _seen_labels=8,
                _tag='valid', _metrics={'Dice': torch.tensor(0.42)})

    payload = fake_wandb.log.call_args[0][0]
    # Tensor should have been coerced to a plain Python float.
    assert payload['valid/Dice'] == pytest.approx(0.42)
    assert isinstance(payload['valid/Dice'], float)


# --- MetricLogger fan-out --------------------------------------------------

def test_metric_logger_forwards_to_wandb_when_provided(fake_wandb):
    fake_wandb.run = MagicMock()

    from src.utils.metrics.log.metric_logger import MetricLogger
    from src.utils.metrics.log.metric_wandb import MetricWandb

    wb = MetricWandb()
    logger = MetricLogger(wb)
    logger.log(_epoch=1, _step=10, _seen_labels=80, _tag='train',
               _metrics={'Loss': 0.1})

    assert fake_wandb.log.called


def test_metric_logger_works_without_wandb_backend():
    """Backwards compat: MetricLogger constructed without _metric_wandb."""
    from src.utils.metrics.log.metric_logger import MetricLogger

    logger = MetricLogger()
    # Should not raise.
    logger.log(_epoch=0, _step=1, _seen_labels=8, _tag='train',
               _metrics={'Loss': 0.5})


# --- Snapper artifact upload ----------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_snapper_save_uploads_artifact_when_wandb_run_active(fake_wandb, tmp_snapshot_dir):
    fake_wandb.run = MagicMock()  # active run

    from src.train.snapper import Snapper
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=0, _step=1, _seen_label=8, _async=False)

    assert fake_wandb.save.called
    save_path = fake_wandb.save.call_args[0][0]
    assert save_path.endswith('000-0001.pt')


def test_snapper_save_skips_upload_when_no_active_run(fake_wandb, tmp_snapshot_dir):
    fake_wandb.run = None  # no active run

    from src.train.snapper import Snapper
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=0, _step=1, _seen_label=8, _async=False)

    assert not fake_wandb.save.called


# --- maybe_init_wandb ------------------------------------------------------

def test_maybe_init_wandb_disabled_returns_false():
    from train import maybe_init_wandb

    configs = {'trainer': {'wandb': {'enabled': False}}}
    assert maybe_init_wandb(configs) is False


def test_maybe_init_wandb_enabled_calls_wandb_init(fake_wandb):
    from train import maybe_init_wandb

    configs = {
        'trainer': {
            'wandb': {
                'enabled': True,
                'project': 'gbm-seg',
                'entity': 'someone',
                'run_name': 'test-run',
            },
        },
    }
    assert maybe_init_wandb(configs, _fold=2) is True
    fake_wandb.init.assert_called_once()
    init_kwargs = fake_wandb.init.call_args.kwargs
    assert init_kwargs['project'] == 'gbm-seg'
    assert init_kwargs['entity'] == 'someone'
    assert init_kwargs['name'] == 'test-run'
    # The full configs dict (plus fold) is logged as wandb.config so every
    # ablation axis is filterable on the W&B UI.
    assert init_kwargs['config']['fold'] == 2
    assert init_kwargs['config']['trainer']['wandb']['enabled'] is True
