"""DDP plumbing helpers.

DistributedDataParallel is opt-in via ``trainer.runtime.ddp: True`` in the
experiment yaml. When enabled, training must be launched under
``torchrun`` (or another launcher that sets ``LOCAL_RANK``/``RANK``/
``WORLD_SIZE`` env vars). ``sbatch/train_ddp.sbatch`` wraps this for
SLURM single-node 4-GPU launches.

DataParallel (``trainer.runtime.dp: True``) and DDP are mutually exclusive —
DDP takes precedence when both flags are set, and the DP path remains
fully functional for models/runs that aren't ready for DDP yet
(notably the 3D U-Net continues to use DP).

Modules that fan out across ranks (factory, trainer, snapper,
metric_logger, metric_wandb, train.maybe_init_wandb,
GPURunningMetrics) talk to torch.distributed only through these
helpers, so the rank-0-only paths and the all-reduce patterns stay
isolated and easy to test.
"""

import logging
import os

import torch
import torch.distributed as dist


def ddp_requested(_configs: dict) -> bool:
    """True iff the experiment config asks for DDP."""
    trainer = _configs.get('trainer', {}) or {}
    runtime = trainer.get('runtime', {}) or {}
    return bool(runtime.get('ddp', False))


def ddp_launchable() -> bool:
    """True iff the env vars `torchrun` sets are present.

    Without these we cannot bring up a process group — the caller falls
    back to single-process mode (DP or unwrapped).
    """
    return all(k in os.environ for k in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'))


def init_ddp(_configs: dict):
    """Initialise the NCCL process group when DDP is requested + launchable.

    Returns ``(local_rank, world_size)``. When DDP isn't active, returns
    ``(0, 1)`` so callers can use the same plumbing for both paths.
    """
    if not (ddp_requested(_configs) and ddp_launchable()):
        return 0, 1
    if dist.is_initialized():
        return get_local_rank(), get_world_size()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logging.info(
        "DDP initialised: global_rank=%d local_rank=%d world_size=%d "
        "device=cuda:%d",
        rank, local_rank, world_size, local_rank)
    return local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', '0'))


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """rank-0 in distributed mode; always True in single-process mode."""
    return get_rank() == 0


def all_reduce_sum_(_tensor: torch.Tensor) -> torch.Tensor:
    """Sum-reduce a tensor across all ranks (in-place). No-op when not
    distributed."""
    if is_distributed():
        dist.all_reduce(_tensor, op=dist.ReduceOp.SUM)
    return _tensor
