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
import torch

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


def test_offline_aug_variants_are_actually_used(tmp_path):
    """Regression for the pre-refactor bug where offline-aug variants were
    loaded into self.images under a suffixed key but never returned by
    __getitem__ — because image_list got the BASE filename, so the variant
    indices' file_name lookup hit the original volume's entry. The fix is
    in src/data/ds_train.py: image_list now stores the actual file path
    (original OR cache file), and __getitem__ lazy-loads that path.

    This test exercises both index ranges and asserts the returned
    patches differ (they would have been bit-identical under the bug).
    """
    d = _make_ds_train(tmp_path, _n=1)
    _build(d, _precompute=True)                # populate the twist cache
    ds = GBMDataset(                           # training mode, no online aug
        _source_directory=d,
        _sample_dimension=SAMPLE_DIM,
        _pixel_per_step=STRIDE,
        _ignore_stride_mismatch=True,
        _augmentation_offline=OFFLINE,
        _augmentation_workers=1,
        _offline_precompute=False)             # _augmentation_online=None by default
    # Two entries: original (indices 0..N-1) + twist variant (N..2N-1).
    # twist preserves shape, so samples_per_image is identical for both.
    n_per = ds.samples_per_image[0]
    assert ds.samples_per_image[1] == n_per
    orig = ds[0]['sample']
    variant = ds[n_per]['sample']
    # _twist_clock rotates each Z slice by a per-slice angle (~±4° here on
    # a 16-deep volume with step 0.5°), which perturbs pixel values almost
    # everywhere on the random-valued test images. If the variant were
    # silently returning the original (the pre-fix bug), these would be
    # bit-identical and torch.equal would return True.
    assert not torch.equal(orig, variant), \
        "variant-range index should yield augmented data, not the original"
