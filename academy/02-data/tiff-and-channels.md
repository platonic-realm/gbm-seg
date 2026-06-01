# TIFF layout and channels

The data is stored as multi-page TIFF files. PyTorch's view of a 4D
array is `(Z, C, H, W)` — Z slices, channels, height, width. The TIFFs
on disk store the same data in **ImageJ's `ZCYX` order**, which means
the `(Z, C, H, W)` interpretation is correct out of the box from
`tifffile.imread`.

## Channels

Each training TIFF has **4 channels**:

| Channel index | Stain / role | What it shows |
|---|---|---|
| 0 | nephrin (fluorescent stain) | Urine-side podocyte foot processes |
| 1 | collagen-IV (fluorescent stain) | GBM scaffold |
| 2 | WGA (fluorescent stain) | Glycoprotein-rich surfaces |
| 3 | label (binary mask) | Expert annotation: 1 = GBM, 0 = background |

Test volumes in `ds_test_unlabeled/` only have channels 0-2 (no label
column). Test volumes in `ds_test_labeled/` are arranged as one
sub-directory per annotator (Chris, David, Robin), each with its own
4-channel TIFFs — same images, different label channels reflecting each
annotator's GBM tracing. We use these to measure
[inter-rater agreement](../06-inference/expert-comparison.md).

## The 3-channel input rule

The model always takes **3 input channels** (nephrin, collagen-4, WGA).
The label channel is the *target*, not an input. This is why
`InferenceDataset.getNumberOfChannels` returns `3` regardless of the
input TIFF's actual channel count — see
`src/data/ds_infer.py:134`.

This convention exists because the labelled test crops have 4 channels
(label is the 4th), but the model trained on training data also with 4
channels was constructed with `in_channels=3`. Without the hard-coded
3, the factory would try to build a 4-channel model for the labelled
test crops and crash with a `patch_embed.proj.weight` size mismatch at
snapshot-load time. See the `ds_infer.getNumberOfChannels` commit
(`b254d89`) and [inference-patch-must-match-train](../07-case-study-z-jaggedness/diagnosis.md)
for the broader pattern (the trained shape is baked into the model in
multiple ways, not just the input channel count).

## Where to read the TIFFs

- Source dataset (pre-create) — `<default_data_path>/ds_train/*.tif`,
  also `ds_test_unlabeled/` and `ds_test_labeled/<annotator>/`. The
  `default_data_path` is read from
  `configs/template.yaml: experiments.default_data_path`.
- Experiment-local (post-create, **resized + Z-upsampled**) —
  `<experiments_root>/<exp>/datasets/ds_train/*.tiff`. These are big
  (BigTIFF when over 4 GB). Don't open them by reflex in Fiji; they're
  meant for the model.

## A worked example

A typical raw training TIFF before `gbm.py create`:
- Shape `(Z=13, C=4, H=2048, W=2048)`, dtype `uint16`, ~860 MB.
- Voxel size in the TIFF metadata: roughly `(0.05, 0.05, 0.3)` µm (some
  files lack `XResolution`/`YResolution` — see
  `src/utils/misc.py:get_voxel_size` for the soft fallback).

After `gbm.py create`:
- XY zoomed to match the target voxel size — usually a no-op (the
  source is already at 0.05 µm/pixel), but a few files at coarser
  native XY are zoomed up.
- Z upsampled 6× via trilinear interpolation on channels 0-2 and
  `np.repeat` on the label channel — see [Z anisotropy and
  label upsampling](z-anisotropy-and-upsampling.md).
- Result shape `(Z=78, C=4, H=2164, W=2164)`, dtype `float32`, ~5 GB
  per file (LZW compression in the output TIFF).
