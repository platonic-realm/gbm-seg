"""Validation metric masking — restrict to real-label Z slices only.

When the training data has been Z-interpolated and labels stacked (each
manually-drawn slice tiled to fill the interpolated z-positions), the
labels at non-original Z indices are just copies of the previous real
slice. Computing validation metrics over every Z slice inflates them
by counting the same label voxel up to ``scale_factor`` times.

These tests cover ``Unet3DTrainer._select_real_z`` in isolation, plus
the path from ``z_start`` through to the masked confusion matrix that
``Metrics`` would build.
"""

import torch

from src.train.trainer import Unet3DTrainer


def _make_trainer(stride: int) -> Unet3DTrainer:
    t = Unet3DTrainer.__new__(Unet3DTrainer)
    t.valid_label_stride = int(stride)
    return t


def test_select_real_z_passthrough_when_stride_one():
    t = _make_trainer(stride=1)
    pred = torch.zeros(2, 12, 4, 4, dtype=torch.long)
    label = torch.ones(2, 12, 4, 4, dtype=torch.long)
    z_start = torch.tensor([0, 3])
    out_pred, out_label = t._select_real_z(pred, label, z_start)
    assert out_pred is pred
    assert out_label is label


def test_select_real_z_passthrough_when_z_start_missing():
    t = _make_trainer(stride=6)
    pred = torch.zeros(2, 12, 4, 4, dtype=torch.long)
    label = torch.ones(2, 12, 4, 4, dtype=torch.long)
    out_pred, out_label = t._select_real_z(pred, label, None)
    assert out_pred is pred and out_label is label


def test_select_real_z_picks_every_sixth_slice_z_start_zero():
    """sample_dim Z=12, stride=6, z_start=0 → offsets 0, 6 are real."""
    t = _make_trainer(stride=6)
    pred = torch.arange(12).view(1, 12, 1, 1).expand(1, 12, 4, 4).clone()
    label = pred.clone()
    z_start = torch.tensor([0])
    out_pred, out_label = t._select_real_z(pred, label, z_start)
    # 2 slices kept; first plane should be slice 0, second plane should be slice 6.
    assert out_pred.shape == (2, 4, 4)
    assert out_label.shape == (2, 4, 4)
    assert torch.all(out_pred[0] == 0)
    assert torch.all(out_pred[1] == 6)


def test_select_real_z_offsets_shift_with_z_start():
    """z_start=3 with stride=6 → offsets 3, 9 (so absolute z = 6, 12)."""
    t = _make_trainer(stride=6)
    pred = torch.arange(12).view(1, 12, 1, 1).expand(1, 12, 4, 4).clone()
    label = pred.clone()
    z_start = torch.tensor([3])
    out_pred, _ = t._select_real_z(pred, label, z_start)
    # offset 3 → slice index 3 in the patch; offset 9 → slice index 9.
    assert out_pred.shape == (2, 4, 4)
    assert torch.all(out_pred[0] == 3)
    assert torch.all(out_pred[1] == 9)


def test_select_real_z_handles_batch_with_different_z_starts():
    """Item 0 has z_start=0 (real at 0,6); item 1 has z_start=1 (real at 5,11)."""
    t = _make_trainer(stride=6)
    pred = torch.arange(12).view(1, 12, 1, 1).expand(2, 12, 2, 2).clone()
    label = pred.clone()
    z_start = torch.tensor([0, 1])
    out_pred, _ = t._select_real_z(pred, label, z_start)
    # 4 slices total: item-0 offsets {0,6} + item-1 offsets {5,11}.
    assert out_pred.shape == (4, 2, 2)
    assert torch.all(out_pred[0] == 0)
    assert torch.all(out_pred[1] == 6)
    assert torch.all(out_pred[2] == 5)
    assert torch.all(out_pred[3] == 11)


def test_metrics_on_masked_volume_excludes_stacked_label_voxels():
    """End-to-end: confusion matrix from sliced tensors counts only real slices.

    Construct a tiny "stacked label" example: 12-deep label volume where
    every 6 slices are identical (the stacking pattern). Prediction is
    correct on the real slices and wrong everywhere else. With stride=6
    masking, accuracy should be 100% on the real slices; without masking
    it would be 2/12 = 16.7%.
    """
    from src.utils.metrics.clfication import Metrics

    Z, H, W, C = 12, 2, 2, 2
    label = torch.zeros(1, Z, H, W, dtype=torch.long)
    pred = torch.zeros(1, Z, H, W, dtype=torch.long)
    # Real label slices (z % 6 == 0): all-foreground.
    for z in (0, 6):
        label[0, z] = 1
    # Stacked copies at z=1..5 and z=7..11 mirror the previous real slice.
    for z in range(1, 6):
        label[0, z] = label[0, 0]
    for z in range(7, 12):
        label[0, z] = label[0, 6]
    # Prediction is correct on real slices only.
    for z in (0, 6):
        pred[0, z] = 1
    # Wrong (all background) on stacked copies.

    # Unmasked: Dice over the full patch — wrong on 10/12 slices.
    m_full = Metrics(C, pred, label)
    dice_full = float(m_full.Dice(_class_id=1))

    # Masked: only the 2 real slices counted.
    t = _make_trainer(stride=6)
    p, l = t._select_real_z(pred, label, torch.tensor([0]))
    m_real = Metrics(C, p, l)
    dice_real = float(m_real.Dice(_class_id=1))

    assert dice_full < 0.5, f"Unmasked Dice should be poor: got {dice_full}"
    assert dice_real == 1.0, f"Masked Dice should be perfect: got {dice_real}"


def test_dataset_returns_z_start_in_validation_patch(tmp_path):
    """GBMDataset.__getitem__ must surface z_start so validStep can mask."""
    import numpy as np
    import tifffile

    from src.data.ds_train import GBMDataset

    arr = np.zeros((12, 4, 256, 256), dtype=np.float32)
    arr[:, :3] = 100
    arr[:, 3] = 1
    tifffile.imwrite(tmp_path / "tiny.tiff", arr,
                     shape=arr.shape, imagej=True)

    ds = GBMDataset(_source_directory=str(tmp_path),
                    _sample_dimension=[12, 256, 256],
                    _pixel_per_step=[1, 64, 64],
                    _ignore_stride_mismatch=True)
    sample = ds[0]
    assert 'z_start' in sample
    assert isinstance(sample['z_start'], int)
    assert sample['z_start'] == 0  # only one patch fits at sample_dim
