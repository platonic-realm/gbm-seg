"""Pin the DDP helper surface + rank-0 gating across the project.

We can't bring up a real NCCL process group in a CPU-only pytest run, so
we mock ``torch.distributed`` to simulate the "is_initialized → True"
state and verify each consumer of the helpers gates correctly.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# --- distributed helpers (pure functions) ----------------------------------


def test_ddp_requested_reads_trainer_ddp_flag():
    from src.train.distributed import ddp_requested
    assert not ddp_requested({})
    assert not ddp_requested({'trainer': {}})
    assert not ddp_requested({'trainer': {'runtime': {}}})
    assert not ddp_requested({'trainer': {'runtime': {'ddp': False}}})
    assert ddp_requested({'trainer': {'runtime': {'ddp': True}}})


def test_ddp_launchable_detects_torchrun_envvars(monkeypatch):
    from src.train.distributed import ddp_launchable
    for k in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'):
        monkeypatch.delenv(k, raising=False)
    assert not ddp_launchable()

    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '4')
    assert ddp_launchable()


def test_helpers_single_process_defaults():
    """No active group → rank 0, world 1, main process is True."""
    from src.train.distributed import (
        get_local_rank,
        get_rank,
        get_world_size,
        is_distributed,
        is_main_process,
    )
    assert is_distributed() is False
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True
    # local_rank reads from env; without LOCAL_RANK set defaults to 0.
    if 'LOCAL_RANK' in os.environ:
        del os.environ['LOCAL_RANK']
    assert get_local_rank() == 0


def test_all_reduce_sum_is_noop_in_single_process():
    import torch

    from src.train.distributed import all_reduce_sum_
    t = torch.tensor([1.0, 2.0, 3.0])
    out = all_reduce_sum_(t)
    assert out is t  # same object
    assert (out == torch.tensor([1.0, 2.0, 3.0])).all()


# --- rank-0 gating in the consumers ---------------------------------------


def test_metric_logger_skips_when_not_main_process():
    """Under DDP, only rank-0 should log to stdout + W&B."""
    from src.utils.metrics.log.metric_logger import MetricLogger

    fake_backend = MagicMock()
    logger = MetricLogger(fake_backend)

    with patch('src.train.distributed.is_main_process', return_value=False):
        logger.log(_epoch=0, _samples=100, _tag='train', _metrics={'Dice': 0.5})
    assert not fake_backend.log.called

    with patch('src.train.distributed.is_main_process', return_value=True):
        logger.log(_epoch=0, _samples=100, _tag='train', _metrics={'Dice': 0.5})
    assert fake_backend.log.called


def test_snapper_save_skips_when_not_main_process(tmp_path):
    """Snapper writes from rank-0 only."""
    from torch import nn

    from src.train.snapper import Snapper

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

    snap_dir = tmp_path / "snapshots"
    snapper = Snapper(str(snap_dir))

    with patch('src.train.distributed.is_main_process', return_value=False):
        snapper.save(_Tiny(), _epoch=0, _step=1, _async=False)
    # No file should have been written.
    pts = list(snap_dir.glob('*.pt'))
    assert pts == []

    # rank-0 writes normally.
    with patch('src.train.distributed.is_main_process', return_value=True):
        snapper.save(_Tiny(), _epoch=0, _step=1, _async=False)
    assert (snap_dir / "000-0001.pt").exists()


def test_maybe_init_wandb_skips_when_not_main_process():
    """W&B init only on rank-0; other ranks return False without trying."""
    from train import maybe_init_wandb
    cfg = {'trainer': {'wandb': {'enabled': True}}}
    with patch('train.is_main_process', return_value=False):
        assert maybe_init_wandb(cfg) is False


# --- GPURunningMetrics all-reduce path ------------------------------------


def test_gpu_running_metrics_calls_all_reduce_when_distributed():
    """`calculate` must all-reduce values + counter before dividing
    so each rank reports the global mean."""
    import torch

    from src.utils.metrics.memory import GPURunningMetrics

    m = GPURunningMetrics('cpu', ['Dice', 'Loss'])
    m.add({'Dice': torch.tensor(0.5), 'Loss': torch.tensor(0.1)})

    calls = []

    def fake_all_reduce(tensor):
        calls.append(tensor.clone())
        return tensor

    with patch('src.utils.metrics.memory.is_distributed', return_value=True), \
         patch('src.utils.metrics.memory.all_reduce_sum_',
               side_effect=fake_all_reduce):
        out = m.calculate()

    # Should have all-reduced both `values` and `counter` (2 calls).
    assert len(calls) == 2
    # And produced the correct local mean (since fake_all_reduce is a no-op).
    assert abs(out['Dice'] - 0.5) < 1e-6
    assert abs(out['Loss'] - 0.1) < 1e-6


def test_gpu_running_metrics_skips_all_reduce_when_not_distributed():
    """In single-process mode the all-reduce branch is never taken."""
    import torch

    from src.utils.metrics.memory import GPURunningMetrics

    m = GPURunningMetrics('cpu', ['Dice'])
    m.add({'Dice': torch.tensor(0.8)})

    with patch('src.utils.metrics.memory.is_distributed', return_value=False), \
         patch('src.utils.metrics.memory.all_reduce_sum_') as fake_reduce:
        m.calculate()
    assert not fake_reduce.called


# --- Orchestrator refuses DDP --------------------------------------------


def test_main_train_all_folds_refuses_ddp():
    from train import main_train_all_folds
    cfg = {
        'trainer': {
            'runtime': {'ddp': True},
            'logging': {'wandb': {'enabled': False}},
        },
        'root_path': '/tmp/x',
    }
    with pytest.raises(RuntimeError, match=r"per-fold|torchrun|sbatch"):
        main_train_all_folds(cfg)


def test_main_train_refuses_ddp_without_torchrun_env(monkeypatch):
    """trainer.ddp=True without torchrun env → loud failure with hint."""
    from train import main_train
    for k in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'):
        monkeypatch.delenv(k, raising=False)
    # Build the most minimal config that gets past the early mutations.
    cfg = {
        'trainer': {
            'runtime': {'ddp': True},
            'logging': {
                'snapshot_path': './snapshots/',
                'visualization': {'path': './visuals/', 'enabled': False},
            },
        },
        'logging': {'log_file': 'logs/train.log', 'log_summary': False},
    }
    with pytest.raises(RuntimeError, match=r"torchrun"):
        main_train(cfg, _fold=0)


def test_init_ddp_uses_30min_timeout_by_default(monkeypatch):
    """The default NCCL collective timeout is 30 min (not PyTorch's 10 min)."""
    import datetime
    from unittest.mock import patch

    monkeypatch.delenv('TORCH_NCCL_TIMEOUT_MIN', raising=False)
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '4')

    from src.train import distributed
    cfg = {'trainer': {'runtime': {'ddp': True}}}

    with patch('torch.cuda.set_device'), \
         patch.object(distributed.dist, 'init_process_group') as init_pg, \
         patch.object(distributed.dist, 'get_world_size', return_value=4), \
         patch.object(distributed.dist, 'get_rank', return_value=0), \
         patch.object(distributed.dist, 'is_initialized', return_value=False):
        distributed.init_ddp(cfg)
        init_pg.assert_called_once()
        kwargs = init_pg.call_args.kwargs
        assert kwargs['backend'] == 'nccl'
        assert kwargs['timeout'] == datetime.timedelta(minutes=30)


def test_init_ddp_timeout_overridable_via_env(monkeypatch):
    """`TORCH_NCCL_TIMEOUT_MIN` env var overrides the 30-min default."""
    import datetime
    from unittest.mock import patch

    monkeypatch.setenv('TORCH_NCCL_TIMEOUT_MIN', '60')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '4')

    from src.train import distributed
    cfg = {'trainer': {'runtime': {'ddp': True}}}

    with patch('torch.cuda.set_device'), \
         patch.object(distributed.dist, 'init_process_group') as init_pg, \
         patch.object(distributed.dist, 'get_world_size', return_value=4), \
         patch.object(distributed.dist, 'get_rank', return_value=0), \
         patch.object(distributed.dist, 'is_initialized', return_value=False):
        distributed.init_ddp(cfg)
        assert init_pg.call_args.kwargs['timeout'] == \
            datetime.timedelta(minutes=60)
