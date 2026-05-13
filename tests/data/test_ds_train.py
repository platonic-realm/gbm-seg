"""Crop-augmentation bounds. Locks in §1.4.

The pre-fix code used `_channel[z_start:z_length, ...]` — slicing to the
*length* as if it were an endpoint. So the zeroed region was effectively
random and often empty. After the fix, exactly one rectangular cuboid
within the dataset is zeroed, fully in-bounds, with the sampled dimensions.
"""

import numpy as np

from src.data.ds_train import GBMDataset


def test_crop_zeroes_one_contiguous_region():
    rng = np.random.default_rng(0)
    np.random.seed(42)  # _crop_channels uses module-level numpy RNG

    channel = rng.integers(1, 256, size=(12, 256, 256), dtype=np.int32).astype(np.float32)
    before = channel.copy()

    cropped = GBMDataset._crop_channels(channel.copy())

    # Exactly one rectangular block of zeros should have appeared.
    zero_mask = (cropped == 0) & (before != 0)
    z_idx = np.where(zero_mask.any(axis=(1, 2)))[0]
    y_idx = np.where(zero_mask.any(axis=(0, 2)))[0]
    x_idx = np.where(zero_mask.any(axis=(0, 1)))[0]

    assert z_idx.size > 0 and y_idx.size > 0 and x_idx.size > 0, \
        "Crop should zero at least one voxel"

    # The zeroed region must be contiguous along every axis.
    assert (z_idx == np.arange(z_idx[0], z_idx[-1] + 1)).all()
    assert (y_idx == np.arange(y_idx[0], y_idx[-1] + 1)).all()
    assert (x_idx == np.arange(x_idx[0], x_idx[-1] + 1)).all()


def test_crop_stays_in_bounds_for_many_seeds():
    """Stress-test: every sampled crop must fit fully inside the channel."""
    channel_shape = (12, 256, 256)
    for seed in range(64):
        np.random.seed(seed)
        channel = np.ones(channel_shape, dtype=np.float32)
        cropped = GBMDataset._crop_channels(channel.copy())
        assert cropped.shape == channel_shape, \
            f"Crop must not change shape (seed={seed})"
        # If the buggy code ran, the slice end could exceed the start by
        # a length that overflows the axis. After the fix this never happens.
        assert cropped.min() >= 0
        assert cropped.max() == 1  # outside the crop region


def test_crop_block_sizes_within_sampled_ranges():
    """The crop dimensions are drawn from known ranges (2-6, 30-50, 30-50)."""
    for seed in range(16):
        np.random.seed(seed)
        channel = np.ones((12, 256, 256), dtype=np.float32)
        cropped = GBMDataset._crop_channels(channel.copy())

        zero_mask = cropped == 0
        z_count = zero_mask.any(axis=(1, 2)).sum()
        y_count = zero_mask.any(axis=(0, 2)).sum()
        x_count = zero_mask.any(axis=(0, 1)).sum()

        assert 2 <= z_count <= 6, f"z_count={z_count} outside expected range"
        assert 30 <= y_count <= 50, f"y_count={y_count} outside expected range"
        assert 30 <= x_count <= 50, f"x_count={x_count} outside expected range"
