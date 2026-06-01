# Data overview

This directory explains how the dataset is shaped, what happens to it
during `gbm.py create`, and how it's consumed by the training and
inference loops.

## The big picture

Source files on disk: 4-channel ZCYX TIFFs at the native microscope
sampling. `gbm.py create` makes an experiment-local copy after two
transformations: **XY resize** (so every voxel is at the target lateral
voxel size, `[0.050, 0.050, 0.300]` µm by default) and **Z upsampling**
(by `default_z_scale=6`, so cubic-ish voxels). The resized TIFFs are
~5–30× larger than the source files and live under
`<exp>/datasets/ds_train/`, `<exp>/datasets/ds_test_unlabeled/`, and
`<exp>/datasets/ds_test_labeled/`.

The training dataset reads these resized TIFFs lazily (so a worker only
holds one volume in memory at a time) and emits 3D patches (Z=12, X=256,
Y=256 by default).

## Files in this section

- **[TIFF layout and channels](tiff-and-channels.md)** — the ZCYX axis
  order and what each channel represents.
- **[Z anisotropy and label upsampling](z-anisotropy-and-upsampling.md)**
  — why we upsample Z 6× and what's wrong with `np.repeat` on labels (the
  trigger for [the Z-jaggedness case study](../07-case-study-z-jaggedness/README.md)).
- **[Augmentation](augmentation.md)** — offline (precomputed cache of
  zoom/twist variants) and online (random rotate, blur, crop,
  channel-drop).
- **[Lazy loading and the block-random sampler](lazy-loading-sampler.md)**
  — why volumes are not all loaded into RAM, and the sampler that groups
  all patches from one file together to keep I/O sane.

## Key code paths

- `src/utils/misc.py:resize_and_copy` — the per-TIFF resize + Z-upsample
  function called from `gbm.py create`. Now parallelised across a Pool
  (commit `8b1d034`).
- `src/utils/misc.py:_z_interpolate_and_stack` — the Z-upsampling step.
  Channels are trilinearly interpolated; labels are `np.repeat`'d. See
  [Z anisotropy](z-anisotropy-and-upsampling.md) for the consequences.
- `src/data/ds_base.py:BaseDataset` — abstract base that defines the
  patch tiling (`sample_dimension`, `pixel_stride`).
- `src/data/ds_train.py:GBMDataset` — the training dataset.
- `src/data/ds_infer.py:InferenceDataset` — the inference dataset
  (single volume, no labels in the input even when a 4th label channel
  is present in the TIFF; see `getNumberOfChannels` which hard-returns
  3 — see [TIFF and channels](tiff-and-channels.md) for the why).
- `src/data/samplers.py:FileBlockRandomSampler` — the
  one-file-at-a-time sampler used by `DataLoader` during training.
