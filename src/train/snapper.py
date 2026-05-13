# Python Imports
import glob
import logging
import os
import threading
from typing import Optional

# Library Imports
import torch
from torch.nn import DataParallel, Module

# Local Imports
from src.utils.misc import create_dirs_recursively


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
            save_path = \
                os.path.join(self.snapshot_path,
                             f"{_epoch:03d}-{_step:04d}.pt")
            torch.save(snapshot, save_path)
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
