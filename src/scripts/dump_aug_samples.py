"""Dump before/after-rotation augmentation samples for visual verification.

Loads an experiment's resized + Z-upsampled ``ds_train``, extracts a few
patches that contain GBM foreground, applies the online rotation
augmentation (``GBMDataset._rotate_channels``), and writes before/after
TIFF pairs so the rotation — and the binary-label preservation from the
order=0 fix — can be checked by eye in ImageJ.

Each TIFF is a (Z, C, H, W) ImageJ stack: channels 0-2 = nephrin /
collagen-4 / WGA, channel 3 = the label (scaled to 0/255 for visibility).

Usage:
    python src/scripts/dump_aug_samples.py <experiment_path> [n_samples]
"""

# Python Imports
import glob
import os
import sys

# Library Imports
import numpy as np
import tifffile
import yaml

# Local Imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.ds_train import GBMDataset  # noqa: E402


def _find_foreground_patch(_img, _rng, _sample_dim):
    """Pick a (Z, C, H, W) patch whose label channel has GBM voxels."""
    z_max = _img.shape[0] - _sample_dim[0]
    x_max = _img.shape[2] - _sample_dim[1]
    y_max = _img.shape[3] - _sample_dim[2]
    patch = None
    for _ in range(300):
        z = int(_rng.integers(0, z_max + 1))
        x = int(_rng.integers(0, x_max + 1))
        y = int(_rng.integers(0, y_max + 1))
        patch = _img[z:z + _sample_dim[0], :,
                     x:x + _sample_dim[1], y:y + _sample_dim[2]]
        if patch[:, 3].sum() > 0:
            return patch
    return patch  # fallback: last patch even if empty


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python src/scripts/dump_aug_samples.py "
            "<experiment_path> [n_samples]")
    exp_path = sys.argv[1]
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    ds_train = os.path.join(exp_path, 'datasets', 'ds_train')
    out_dir = os.path.join(exp_path, 'aug_samples')
    os.makedirs(out_dir, exist_ok=True)

    # Patch size = the experiment's training sample_dimension, so the dump
    # always matches what training actually feeds the model.
    with open(os.path.join(exp_path, 'configs.yaml'), encoding='UTF-8') as f:
        sample_dim = tuple(
            yaml.safe_load(f)['trainer']['train_ds']['sample_dimension'])

    tiffs = sorted(glob.glob(os.path.join(ds_train, '*.tif*')))
    if not tiffs:
        raise SystemExit(f"No TIFFs found in {ds_train}")

    rng = np.random.default_rng(0)
    print(f"Dumping {n_samples} before/after rotation pairs (patch "
          f"{sample_dim}) to {out_dir}\n")
    for k in range(n_samples):
        tiff = tiffs[k % len(tiffs)]
        img = tifffile.imread(tiff).astype(np.float32)   # (Z, C, H, W)
        patch = _find_foreground_patch(img, rng, sample_dim)

        nephrin = patch[:, 0].copy()
        collagen4 = patch[:, 1].copy()
        wga = patch[:, 2].copy()
        label = (patch[:, 3] > 0).astype(np.float32)     # binarise to {0,1}

        before = np.stack([nephrin, collagen4, wga, label * 255.0], axis=1)
        tifffile.imwrite(os.path.join(out_dir, f"sample_{k}_before.tiff"),
                         before.astype(np.float32), imagej=True)

        rot_n, rot_c, rot_w, rot_l = GBMDataset._rotate_channels(
            nephrin, collagen4, wga, label)
        after = np.stack([rot_n, rot_c, rot_w, rot_l * 255.0], axis=1)
        tifffile.imwrite(os.path.join(out_dir, f"sample_{k}_after.tiff"),
                         after.astype(np.float32), imagej=True)

        uniq = np.unique(rot_l).tolist()
        print(f"sample {k}: {os.path.basename(tiff)}")
        print(f"  label foreground voxels: before={int(label.sum())} "
              f"after={int(rot_l.sum())}")
        print(f"  rotated-label unique values: {uniq}  "
              f"({'BINARY ok' if set(uniq).issubset({0.0, 1.0}) else 'NOT BINARY'})")

    print(f"\nDone. Open the before/after pairs in {out_dir} with ImageJ.")


if __name__ == '__main__':
    main()
