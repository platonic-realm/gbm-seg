"""Online augmentation — `GBMDataset._rotate_channels`.

Locks in the interpolation-order fix: rotating the binary label must use
nearest-neighbour (order=0) so it stays binary. With order>=1 the label
blends into fractional values that the trainer's `labels.long()` then
truncates to 0, silently eroding the thin GBM membrane on every rotated
patch. The three-axis rotation itself is intentional (it propagates the
high-res XY information into the low-res Z direction) and is NOT changed.
"""

import numpy as np

from src.data.ds_train import GBMDataset


def _binary_label(shape):
    label = np.zeros(shape, dtype=np.float32)
    label[3:9, 16:48, 16:48] = 1.0
    return label


def test_rotate_channels_keeps_label_binary():
    """After online rotation the label contains only {0, 1} — order=0."""
    np.random.seed(1)  # non-zero rotation angles, deterministically
    shape = (12, 64, 64)
    rng = np.random.default_rng(0)
    nephrin = rng.random(shape, dtype=np.float32)
    collagen4 = rng.random(shape, dtype=np.float32)
    wga = rng.random(shape, dtype=np.float32)
    label = _binary_label(shape)

    _, _, _, rotated_label = GBMDataset._rotate_channels(
        nephrin, collagen4, wga, label)

    uniq = set(np.unique(rotated_label).tolist())
    assert uniq.issubset({0.0, 1.0}), (
        f"rotated label must stay binary (order=0), got values: {uniq}")


def test_rotate_channels_preserves_shape():
    """reshape=False — every channel keeps its (Z, X, Y) patch shape."""
    np.random.seed(2)
    shape = (12, 64, 64)
    zeros = np.zeros(shape, dtype=np.float32)
    out = GBMDataset._rotate_channels(zeros.copy(), zeros.copy(),
                                      zeros.copy(), _binary_label(shape))
    for arr in out:
        assert arr.shape == shape


def test_rotate_channels_label_foreground_survives():
    """A nearest-neighbour rotation relocates the foreground but must not
    annihilate it — the rotated label still has foreground voxels (the
    order=1 bug + labels.long() truncation could erode it to nothing)."""
    np.random.seed(3)
    shape = (12, 64, 64)
    zeros = np.zeros(shape, dtype=np.float32)
    label = _binary_label(shape)
    _, _, _, rotated_label = GBMDataset._rotate_channels(
        zeros.copy(), zeros.copy(), zeros.copy(), label)
    assert rotated_label.sum() > 0, "rotation annihilated the label"
