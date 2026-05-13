# Python Imports
import glob
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Optional

# Library Imports
import torch
import yaml
from torch.nn import DataParallel, Module

# Local Imports
from src.utils.misc import create_dirs_recursively


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


def _build_model_card(epoch: int, step: int, seen_labels: int,
                      snapshot_filename: str, experiment_name: Optional[str]) -> dict:
    """Snapshot-time provenance: who/what/when/how this .pt was produced."""
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = None

    # Cast version objects to str so yaml.safe_dump can serialise them
    # (modern torch returns a TorchVersion subclass, not a plain str).
    return {
        'snapshot_filename': snapshot_filename,
        'experiment_name': experiment_name,
        'epoch': int(epoch),
        'step': int(step),
        'seen_labels': int(seen_labels),
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'torch_version': str(torch.__version__),
        'cuda_version': str(torch.version.cuda) if torch.version.cuda else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'gpu_name': gpu_name,
        'python_version': sys.version.split()[0],
        'torch_initial_seed': int(torch.initial_seed()),
        'git_sha': _git_sha(),
    }


class Snapper:
    """Periodic snapshot save/load. Handles DataParallel-wrapped and bare models symmetrically on both ends."""

    def __init__(self, _snapshot_path: str):
        self.snapshot_path = _snapshot_path
        if self.snapshot_path is not None:
            create_dirs_recursively(self.snapshot_path)

    def save(self,
             _model: Module,
             _epoch: int,
             _step: int,
             _seen_label: int,
             _async: bool = True) -> None:
        """Write a snapshot to ``<snapshot_path>/<epoch:03d>-<step:04d>.pt``."""

        if self.snapshot_path is None:
            return
        snapshot = {}
        snapshot['EPOCHS'] = _epoch
        snapshot['STEP'] = _step
        snapshot['SEEN_LABELS'] = _seen_label
        if isinstance(_model, DataParallel):
            snapshot['MODEL_STATE'] = _model.module.state_dict()
        else:
            snapshot['MODEL_STATE'] = _model.state_dict()

        def save_state():
            stem = f"{_epoch:03d}-{_step:04d}"
            save_path = os.path.join(self.snapshot_path, f"{stem}.pt")
            torch.save(snapshot, save_path)

            # Sibling model card with snapshot provenance. Tries to derive the
            # experiment name from the snapshot_path layout
            # (<root>/<exp>/results-train/snapshots/), falls back to None.
            try:
                experiment_name = os.path.basename(
                    os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(self.snapshot_path)))))
            except Exception:
                experiment_name = None

            card = _build_model_card(
                epoch=_epoch, step=_step, seen_labels=_seen_label,
                snapshot_filename=f"{stem}.pt",
                experiment_name=experiment_name)
            card_path = os.path.join(self.snapshot_path, f"{stem}.yaml")
            try:
                with open(card_path, "w", encoding="UTF-8") as f:
                    yaml.safe_dump(card, f, sort_keys=False)
            except Exception as exc:  # pragma: no cover — best-effort
                logging.warning("Failed to write model card %s: %s", card_path, exc)

            logging.info("Snapshot saved on epoch: %d, step: %d",
                         _epoch,
                         _step)
        if _async:
            thread = threading.Thread(target=save_state)
            thread.start()
        else:
            save_state()

    def load(self,
             _model: Module,
             _device: str,
             _path: Optional[str] = None) -> Optional[tuple[int, int]]:

        if not os.path.exists(self.snapshot_path):
            return None

        if _path is None:
            continue_path = os.path.join(self.snapshot_path, 'continue/')
            snapshot_list = sorted(filter(os.path.isfile,
                                          glob.glob(continue_path + '*')),
                                   reverse=True)

            if len(snapshot_list) <= 0:
                return None

            _path = snapshot_list[0]

        snapshot = torch.load(_path,
                              map_location='cpu',
                              weights_only=True)

        state_dict = snapshot['MODEL_STATE']
        target = _model.module if isinstance(_model, DataParallel) else _model
        target.load_state_dict(state_dict)

        epoch = snapshot['EPOCHS']
        seen_labels = snapshot['SEEN_LABELS']

        return epoch, seen_labels
