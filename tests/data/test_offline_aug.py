"""Offline-augmentation cache: precompute vs read-only behaviour.

The offline-augmentation variants (twist/rotate volumes written to
``<ds_train>/cache/``) are built ONCE by ``gbm.py offline-aug``
(``GBMDataset`` with ``_offline_precompute=True``). Training builds the
dataset with ``_offline_precompute=False`` and must only *read* the
cache — a missing entry raises rather than recomputing, because under
DDP every rank would otherwise recompute the same volumes at once
(N x CPU + N x RAM -> swap death).
"""

import os

import numpy as np
import pytest
import tifffile

from src.data.ds_train import GBMDataset

SAMPLE_DIM = [8, 32, 32]
STRIDE = [8, 32, 32]
OFFLINE = [['_twist_clock', '0.5']]


def _make_ds_train(_root, _n=2):
    """A ds_train directory with _n tiny 4-channel (Z,C,H,W) TIFFs."""
    d = _root / "ds_train"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(_n):
        arr = rng.integers(0, 256, size=(16, 4, 64, 64)).astype(np.float32)
        tifffile.imwrite(d / f"vol_{i}.tiff", arr, imagej=True)
    return str(d)


def _build(_dir, _precompute):
    return GBMDataset(
        _source_directory=_dir,
        _sample_dimension=SAMPLE_DIM,
        _pixel_per_step=STRIDE,
        _ignore_stride_mismatch=True,
        _augmentation_offline=OFFLINE,
        _augmentation_workers=1,
        _offline_precompute=_precompute)


def test_precompute_builds_cache(tmp_path):
    """_offline_precompute=True computes + writes the cache."""
    d = _make_ds_train(tmp_path)
    _build(d, _precompute=True)
    cache = os.path.join(d, "cache")
    cached = [f for f in os.listdir(cache) if f.endswith('.tiff')]
    assert len(cached) == 2, f"expected 2 cached aug volumes, got {cached}"


def test_training_reads_prebuilt_cache(tmp_path):
    """With the cache already built, _offline_precompute=False reads it
    and constructs a working dataset (no raise, no recompute)."""
    d = _make_ds_train(tmp_path)
    _build(d, _precompute=True)               # populate the cache
    ds = _build(d, _precompute=False)          # training mode — reads it
    assert len(ds) > 0


def test_training_raises_when_cache_missing(tmp_path):
    """Training mode with no prebuilt cache must raise — never recompute
    (that is the DDP swap-death path the offline-aug command prevents)."""
    d = _make_ds_train(tmp_path)
    with pytest.raises(FileNotFoundError, match="offline-aug"):
        _build(d, _precompute=False)
