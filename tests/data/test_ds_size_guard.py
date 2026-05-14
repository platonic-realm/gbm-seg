"""Size-guard regression: a TIFF smaller than sample_dim must fail loud.

Pre-fix, ``_prepare_images`` silently produced negative ``steps_per_*``
counts when an image was smaller than ``sample_dim`` (floor-division of
``(102 - 256) // 64 + 1 = -3``). That scrambled ``cumulative_sum`` and
let ``__getitem__`` slice out-of-bounds, surfacing later as a cryptic
collate-time shape mismatch (``[3,12,102,64] vs [3,12,256,256]``).
"""

import numpy as np
import pytest
import tifffile

from src.data.ds_infer import InferenceDataset
from src.data.ds_train import GBMDataset


def _write_tiff(path, shape, channels=4):
    """Write a fake ZCYX TIFF of the requested per-slice (Z, H, W) shape."""
    z, h, w = shape
    arr = np.zeros((z, channels, h, w), dtype=np.float32)
    # Put non-zero values so label_correction_function doesn't lose data.
    arr[:, :3] = 100
    if channels == 4:
        arr[:, 3] = 1
    tifffile.imwrite(path, arr, shape=arr.shape, imagej=True)


def test_train_dataset_rejects_too_small_tiff(tmp_path):
    """Training: TIFF with H=102 < sample_dim X=256 must raise."""
    ds_dir = tmp_path / "ds_train"
    ds_dir.mkdir()
    _write_tiff(ds_dir / "tiny.tiff", shape=(12, 102, 64), channels=4)

    with pytest.raises(ValueError, match=r"smaller than sample_dimension"):
        GBMDataset(
            _source_directory=str(ds_dir),
            _sample_dimension=[12, 256, 256],
            _pixel_per_step=[1, 64, 64],
            _ignore_stride_mismatch=True)


def test_train_dataset_rejects_too_short_z(tmp_path):
    """Same guard fires on Z too."""
    ds_dir = tmp_path / "ds_train"
    ds_dir.mkdir()
    _write_tiff(ds_dir / "short_z.tiff", shape=(6, 256, 256), channels=4)

    with pytest.raises(ValueError, match=r"smaller than sample_dimension"):
        GBMDataset(
            _source_directory=str(ds_dir),
            _sample_dimension=[12, 256, 256],
            _pixel_per_step=[1, 64, 64],
            _ignore_stride_mismatch=True)


def test_train_dataset_accepts_exactly_sized_tiff(tmp_path):
    """A TIFF at exactly sample_dim is fine (single patch)."""
    ds_dir = tmp_path / "ds_train"
    ds_dir.mkdir()
    _write_tiff(ds_dir / "exact.tiff", shape=(12, 256, 256), channels=4)

    ds = GBMDataset(
        _source_directory=str(ds_dir),
        _sample_dimension=[12, 256, 256],
        _pixel_per_step=[1, 64, 64],
        _ignore_stride_mismatch=True)
    assert len(ds) == 1


def test_inference_dataset_rejects_too_small_tiff(tmp_path):
    """Inference: same guard on InferenceDataset."""
    f = tmp_path / "small.tiff"
    # Inference dataset takes 3-channel TIFFs (no labels channel).
    _write_tiff(f, shape=(12, 100, 100), channels=3)

    with pytest.raises(ValueError, match=r"smaller than sample_dimension"):
        InferenceDataset(
            _file_path=str(f),
            _sample_dimension=[12, 256, 256],
            _pixel_per_step=[1, 64, 64],
            _scale_factor=1,
            _interpolate=False,
            _no_of_classes=2)
