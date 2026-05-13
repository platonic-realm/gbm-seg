"""Snapper save/load roundtrip.

Locks in §1.3. The pre-fix ``load()`` unconditionally called
``_model.module.load_state_dict(...)`` which crashed when the model was
not wrapped in DataParallel. The fix handles both wrappings symmetrically.

Also exercises §1.7 weights_only=True.
"""

import os

import torch
from torch import nn
from torch.nn.parallel import DataParallel

from src.train.snapper import Snapper


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def _state_dicts_equal(a, b):
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def test_save_load_no_dp(tmp_snapshot_dir):
    src = _TinyModel()
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(src, _epoch=0, _step=1, _seen_label=8, _async=False)

    saved_path = os.path.join(tmp_snapshot_dir, "000-0001.pt")
    assert os.path.exists(saved_path)

    dst = _TinyModel()
    # Make weights distinct so we can verify the load
    with torch.no_grad():
        for p in dst.parameters():
            p.fill_(0.0)

    snapper.load(dst, _device='cpu', _path=saved_path)
    assert _state_dicts_equal(src.state_dict(), dst.state_dict())


def test_save_dp_load_into_non_dp(tmp_snapshot_dir):
    """The bug we're regressing: save with DP, load into a bare model."""
    src = _TinyModel()
    src_dp = DataParallel(src)
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(src_dp, _epoch=0, _step=1, _seen_label=8, _async=False)

    dst = _TinyModel()
    snapper.load(dst, _device='cpu',
                 _path=os.path.join(tmp_snapshot_dir, "000-0001.pt"))
    assert _state_dicts_equal(src.state_dict(), dst.state_dict())


def test_save_non_dp_load_into_dp(tmp_snapshot_dir):
    """The other direction: bare → DP."""
    src = _TinyModel()
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(src, _epoch=0, _step=1, _seen_label=8, _async=False)

    dst = _TinyModel()
    dst_dp = DataParallel(dst)
    snapper.load(dst_dp, _device='cpu',
                 _path=os.path.join(tmp_snapshot_dir, "000-0001.pt"))
    assert _state_dicts_equal(src.state_dict(), dst.state_dict())


def test_load_missing_snapshot_returns_none(tmp_snapshot_dir):
    snapper = Snapper(tmp_snapshot_dir)
    # No file exists; load() should be a no-op returning None.
    result = snapper.load(_TinyModel(), _device='cpu')
    assert result is None
