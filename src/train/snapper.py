# Python Imports
import os
import threading
import logging
import glob

# Library Imports
import torch
from torch.nn import Module, DataParallel

# Local Imports
from src.utils.misc import create_dirs_recursively


class Snapper():

    def __init__(self, _snapshot_path: str):
        self.snapshot_path = _snapshot_path
        if self.snapshot_path is not None:
            create_dirs_recursively(self.snapshot_path)

    def save(self,
             _model: Module,
             _epoch: int,
             _step: int,
             _seen_label: int,
             _async: bool = True):

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
             _path: str or None = None) -> (int, int):

        if not os.path.exists(self.snapshot_path):
            return

        if _path is None:
            continue_path = os.path.join(self.snapshot_path, 'continue/')
            snapshot_list = sorted(filter(os.path.isfile,
                                          glob.glob(continue_path + '*')),
                                   reverse=True)

            if len(snapshot_list) <= 0:
                return

            _path = snapshot_list[0]

        snapshot = torch.load(_path,
                              map_location='cpu')

        # Check if the state dict was created with data parallelism
        state_dict = snapshot['MODEL_STATE']
        # if 'module' in list(state_dict.keys())[0]:
        #     corrected_state_dict = {}
        #     for k, v in state_dict.items():
        #         name = k[7:]  # remove `module.`
        #         corrected_state_dict[name] = v
        #     _model.load_state_dict(corrected_state_dict)
        # else:
        #     _model.load_state_dict(state_dict)

        _model.module.load_state_dict(state_dict)

        epoch = snapshot['EPOCHS']
        seen_labels = snapshot['SEEN_LABELS']

        return epoch, seen_labels
