# Python Imports
import glob
import logging
import os
import random
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Library Imports
import numpy as np
import torch
from torch.nn import DataParallel, Module

# Local Imports


def _git_sha(cwd: str = None) -> Optional[str]:
    """Return the current git HEAD SHA, or None if not in a git repo."""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL, timeout=2)
        return result.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return None


def _build_model_card(epoch: int, step: int,
                      snapshot_filename: str,
                      experiment_name: Optional[str]) -> dict:
    """Snapshot-time provenance: who/what/when/how this snapshot was produced."""
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = None

    return {
        'snapshot_filename': snapshot_filename,
        'experiment_name': experiment_name,
        'epoch': int(epoch),
        'step': int(step),
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'torch_version': str(torch.__version__),
        'cuda_version': str(torch.version.cuda) if torch.version.cuda else None,
        'cudnn_version': torch.backends.cudnn.version()
                         if torch.backends.cudnn.is_available() else None,
        'gpu_name': gpu_name,
        'python_version': sys.version.split()[0],
        'torch_initial_seed': int(torch.initial_seed()),
        'git_sha': _git_sha(),
    }


def _unwrap(_model: Module) -> Module:
    """Return the inner module, stripping DP / DDP wrappers symmetrically."""
    from torch.nn.parallel import DistributedDataParallel
    if isinstance(_model, (DataParallel, DistributedDataParallel)):
        return _model.module
    return _model


class Snapper:
    """Periodic snapshot save/load.

    A snapshot is a single ``.pt`` file containing everything needed to
    resume training cleanly: model weights, optimiser state, scheduler
    state, AMP scaler state, step counter, best-validation tracker, all
    four RNG states (python/numpy/torch/cuda), the W&B run id (for
    ``wandb.init(resume="must", ...)`` continuation), and a provenance
    card (formerly a sibling ``.yaml`` file).

    Resume is opt-in: a user moves the chosen snapshot into
    ``<snapshot_path>/continue/`` and the next training run picks it up
    on ``Snapper.load(...)``. Files in ``<snapshot_path>/`` itself are
    treated as historical and ignored by ``load()`` unless ``_path`` is
    passed explicitly (used by inference).

    DDP / DP wrappers are handled symmetrically on both ends.
    """

    def __init__(self, _snapshot_path: str):
        self.snapshot_path = _snapshot_path
        if self.snapshot_path is not None:
            # The per-fold path used by train.main_train looks like
            # `.../results-train/snapshots/fold_3` (no trailing slash and
            # not yet existing). Make the directory directly here so the
            # `<snapshot_path>/<epoch>-<step>.pt` save target exists.
            Path(self.snapshot_path).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(self,
             _model: Module,
             _epoch: int,
             _step: int,
             _stepper=None,
             _optimizer=None,
             _scheduler=None,
             _best_valid_metrics: Optional[dict] = None,
             _best_valid_epoch: Optional[int] = None,
             _best_valid_step: Optional[int] = None,
             _async: bool = True) -> None:
        """Write a single ``.pt`` containing everything needed to resume.

        Under DDP only the rank-0 process writes — the other ranks have
        identical weights (DDP keeps them in sync), so saving five copies
        of the same model would just race on the same path.
        """
        from src.train.distributed import is_main_process
        if not is_main_process():
            return
        if self.snapshot_path is None:
            return

        stem = f"{_epoch:03d}-{_step:04d}"
        save_path = os.path.join(self.snapshot_path, f"{stem}.pt")

        # Capture wandb run id if a run is active. Used at resume time
        # to call wandb.init(resume="must", id=...) so the new run
        # appends to the original instead of starting a fresh chart.
        # Constrain to str so a stubbed/mock wandb (e.g. in tests) doesn't
        # poison the snapshot with an unpicklable object.
        wandb_run_id = None
        try:
            import wandb
            if wandb.run is not None and isinstance(wandb.run.id, str):
                wandb_run_id = wandb.run.id
        except ImportError:
            pass

        card = _build_model_card(
            epoch=_epoch, step=_step,
            snapshot_filename=f"{stem}.pt",
            experiment_name=self._infer_experiment_name())

        snapshot = {
            'MODEL_STATE': _unwrap(_model).state_dict(),
            'OPTIMIZER_STATE': (_optimizer.state_dict()
                                if _optimizer is not None else None),
            'SCHEDULER_STATE': (_scheduler.state_dict()
                                if _scheduler is not None else None),
            'STEPPER_STATE': (_stepper.state_dict()
                              if _stepper is not None else None),
            'EPOCH': int(_epoch),
            'STEP': int(_step),
            'BEST_VALID_METRICS': _best_valid_metrics,
            'BEST_VALID_EPOCH': _best_valid_epoch,
            'BEST_VALID_STEP': _best_valid_step,
            'WANDB_RUN_ID': wandb_run_id,
            'RNG_PYTHON': random.getstate(),
            'RNG_NUMPY': np.random.get_state(),
            'RNG_TORCH': torch.get_rng_state(),
            'RNG_TORCH_CUDA': (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
            'MODEL_CARD': card,
        }

        def write():
            torch.save(snapshot, save_path)

            # E2: upload the snapshot as an artifact when a W&B run is active.
            try:
                import wandb
                if wandb.run is not None:
                    wandb.save(save_path,
                               base_path=os.path.dirname(save_path),
                               policy='live')
            except ImportError:
                pass
            except Exception as exc:  # pragma: no cover — networked side-effect
                logging.warning("W&B snapshot upload failed: %s", exc)

            logging.info("Snapshot saved on epoch: %d, step: %d",
                         _epoch, _step)

        if _async:
            threading.Thread(target=write).start()
        else:
            write()

    def _infer_experiment_name(self) -> Optional[str]:
        """Best-effort: pull experiment name from the snapshot-path layout
        ``<root>/<exp>/results-train/snapshots/[fold_N]``."""
        try:
            abs_path = os.path.abspath(self.snapshot_path)
            # Walk up to find a directory named results-train and take its parent.
            parts = abs_path.split(os.sep)
            if 'results-train' in parts:
                idx = parts.index('results-train')
                return parts[idx - 1] if idx > 0 else None
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    def load(self,
             _model: Module,
             _device: str = 'cpu',
             _path: Optional[str] = None,
             _stepper=None,
             _optimizer=None,
             _scheduler=None,
             _restore_rng: bool = True) -> Optional[dict]:
        """Restore training state in-place from a single ``.pt`` snapshot.

        Auto-discovery mode (``_path is None``): looks in
        ``<snapshot_path>/continue/`` for the most-recent ``.pt`` and
        loads it. Files in ``<snapshot_path>/`` itself (historical
        snapshots from the run that wrote them) are intentionally
        ignored — resume is opt-in by moving a chosen snapshot into
        ``continue/``.

        Explicit-path mode (``_path`` given): loads exactly that file.
        Used by inference and tests.

        Components passed as ``None`` are simply not restored — useful
        for inference (model only) and for backwards compatibility with
        older snapshots that only carry ``MODEL_STATE``.

        Returns a dict with the resume information::

            {
                'epoch':       int,        # last saved epoch
                'step':        int,        # last saved global step
                'best_metrics': dict|None,  # best validation tracker
                'best_epoch':   int|None,
                'best_step':    int|None,
                'wandb_run_id': str|None,
            }

        Or ``None`` if no snapshot was found / loaded.
        """
        if self.snapshot_path is not None and not os.path.exists(self.snapshot_path):
            return None

        if _path is None:
            continue_path = os.path.join(self.snapshot_path, 'continue')
            if not os.path.isdir(continue_path):
                return None
            snapshot_list = sorted(
                filter(os.path.isfile,
                       glob.glob(os.path.join(continue_path, '*.pt'))),
                reverse=True)
            if not snapshot_list:
                return None
            _path = snapshot_list[0]

        snapshot = torch.load(_path,
                              map_location=_device,
                              weights_only=False)
        # weights_only=False is required because the snapshot now contains
        # python objects (RNG states are tuples / numpy arrays, optimizer
        # state has python dicts, the model card has datetime strings).
        # Snapshots are produced by this codebase and stored on a trusted
        # cluster filesystem, so the elevated trust is acceptable. Do NOT
        # load untrusted snapshots with this setting.

        # Model weights (always; old snapshots have only this).
        if 'MODEL_STATE' in snapshot:
            target = _unwrap(_model)
            target.load_state_dict(snapshot['MODEL_STATE'])

        # Optimizer / scheduler / stepper — silently skipped if either the
        # component or the snapshot key is missing (backwards compat).
        if _optimizer is not None and snapshot.get('OPTIMIZER_STATE') is not None:
            _optimizer.load_state_dict(snapshot['OPTIMIZER_STATE'])

        if _scheduler is not None and snapshot.get('SCHEDULER_STATE') is not None:
            # When the scheduler type changes between the original run and the
            # resume (e.g. reduce_on_plateau → poly_decay because the user
            # decided to switch strategy for the continuation), the saved
            # state_dict is type-incompatible. Skip rather than crash — the
            # new scheduler starts fresh from its config-driven defaults.
            try:
                _scheduler.load_state_dict(snapshot['SCHEDULER_STATE'])
            except (KeyError, TypeError, ValueError) as exc:
                logging.warning(
                    "Scheduler state in snapshot is incompatible with the "
                    "current scheduler (%s); starting the scheduler fresh. "
                    "Underlying error: %s",
                    type(_scheduler).__name__, exc)

        if _stepper is not None and snapshot.get('STEPPER_STATE') is not None:
            _stepper.load_state_dict(snapshot['STEPPER_STATE'])

        # RNG states — best effort; missing keys silently skipped.
        if _restore_rng:
            if 'RNG_PYTHON' in snapshot:
                try:
                    random.setstate(snapshot['RNG_PYTHON'])
                except Exception as exc:  # pragma: no cover
                    logging.warning("Failed to restore python RNG: %s", exc)
            if 'RNG_NUMPY' in snapshot:
                try:
                    np.random.set_state(snapshot['RNG_NUMPY'])
                except Exception as exc:  # pragma: no cover
                    logging.warning("Failed to restore numpy RNG: %s", exc)
            if 'RNG_TORCH' in snapshot:
                try:
                    torch.set_rng_state(snapshot['RNG_TORCH'])
                except Exception as exc:  # pragma: no cover
                    logging.warning("Failed to restore torch RNG: %s", exc)
            if (snapshot.get('RNG_TORCH_CUDA') is not None
                    and torch.cuda.is_available()):
                try:
                    torch.cuda.set_rng_state_all(snapshot['RNG_TORCH_CUDA'])
                except Exception as exc:  # pragma: no cover
                    logging.warning("Failed to restore cuda RNG: %s", exc)

        # Old snapshots used 'EPOCHS' (plural). New ones use 'EPOCH'.
        # Old snapshots also lack BEST_VALID_* / WANDB_RUN_ID — fields
        # default to None so a downstream resume just lacks those niceties.
        epoch = snapshot.get('EPOCH', snapshot.get('EPOCHS', 0))
        step = snapshot.get('STEP', 0)

        return {
            'epoch': int(epoch),
            'step': int(step),
            'best_metrics': snapshot.get('BEST_VALID_METRICS'),
            'best_epoch': snapshot.get('BEST_VALID_EPOCH'),
            'best_step': snapshot.get('BEST_VALID_STEP'),
            'wandb_run_id': snapshot.get('WANDB_RUN_ID'),
        }
