# Z anisotropy and label upsampling

This is the single most consequential data quirk in the whole project.
Understanding it unlocks the [Z-jaggedness case
study](../07-case-study-z-jaggedness/README.md), the
[Cont loss design](../04-loss-cont.md), and the awkward parts of how
[SwinUNETR](../03-models/swinunetr.md) currently behaves.

## Why the voxels aren't cubes

Confocal microscopes have anisotropic resolution: lateral (XY) is set
by the diffraction limit and is sub-micron; axial (Z) is set by the
point-spread function along the optical axis and is much coarser. In
this dataset the source voxel size is roughly `(0.05, 0.05, 0.3)` µm
(X, Y, Z) — **Z is 6× coarser than XY**.

If you fed the raw stack to a 3D U-Net with isotropic 3×3×3 kernels,
each kernel would cover 0.15×0.15×0.9 µm of physical space — 6× more Z
than XY. That's bad for two reasons:

1. The GBM is ~200–400 nm thick. A 0.3 µm Z spacing means the membrane
   often fits *inside one Z slice*, so adjacent slices show jumps of
   "no GBM" → "GBM" → "no GBM" rather than a smooth profile.
2. The conv inductive bias assumes spatial continuity (smooth across
   neighbouring voxels). Strong anisotropy breaks that assumption
   directionally.

## The fix: upsample Z 6× during `gbm.py create`

`gbm.py create` produces an experiment-local copy of the dataset where
**Z has been upsampled 6×** so the voxels are cubic-ish. Two things
happen, and they're not symmetric:

1. **Intensity channels** (0–2: nephrin/collagen-4/WGA) are upsampled
   via trilinear interpolation. A smooth membrane in physical space
   becomes a smooth signal in the upsampled volume.

2. **The label channel** (3) is upsampled by `np.repeat`. If the
   original label at Z=5 is "1" at some `(x, y)`, then the upsampled
   labels at Z=30, 31, 32, 33, 34, 35 are all "1" at that same `(x, y)`.

   See `src/utils/misc.py:_z_interpolate_and_stack` (the function name
   says "stack" because it's stacking-by-repeat for labels and
   trilinear-by-interpolate for intensities, all in one pass).

## The stacked-label artifact

After upsampling, each "original" Z slice in the label channel has been
replicated `z_scale = 6` times in a row. If you scroll Z in Fiji on the
label channel of an upsampled training TIFF, you see **flat plateaus of
6 identical slices**, separated by sharp transitions wherever the
original annotator's mask changed. The intensity channels in contrast
vary smoothly between those plateaus.

This means the training target has an artificial **period-6 structure
in Z**: a model that wants to minimise the loss can learn "predict
identical labels for 6 Z positions in a row, then sometimes change".
That's a discrete pattern, easy to memorise for a model with the right
inductive bias.

The U-Net's small-kernel conv encoder can't memorise this pattern
cleanly — its receptive field grows slowly in Z and its 3×3×3 kernel
shares weights across every Z position, so it produces broadly smooth Z
output. [SwinUNETR](../03-models/swinunetr.md), via its
[window attention + learned relative-position bias](../03-models/window-attention.md),
*can* memorise it, and does — see [the diagnosis](../07-case-study-z-jaggedness/diagnosis.md).

## Why not just use trilinear-interpolated labels?

That was the obvious-sounding fix; we tried it as a thought experiment
and proved it doesn't work for binary labels. See
`src/scripts/compare_label_upsampling.py` (run with
`srun -p cpu -c 4 --mem=16G -t 0:15:00 …`) — it produces side-by-side
TIFFs. The aggregate stats:

```
metric                    stacked    trilinear
foreground label voxels   21,941,040 21,941,040
Z-transitions in label    12         12
```

The two outputs are byte-identical. The reason: trilinear interpolation
of `{0, 255}` labels at `align_corners=False` followed by thresholding
at `127.5` gives each output voxel the value of its **nearest source
slice**. That's exactly `np.repeat`. The transitions move to different
positions only if the threshold is biased — which would just shift the
stairsteps, not smooth them. Genuine smoothness would require either
**soft (probability) labels** — a substantial loss-function change — or
**distance-transform-based labels** (SDF, threshold at 0). See
[Z-jaggedness future directions](../07-case-study-z-jaggedness/future-directions.md).

## How the model + loss + validation respond

Three pieces of the system are aware of this stacking:

1. **The [Cont loss](../04-loss-cont.md)** has a Z-continuity term that
   softmax-diffs adjacent Z slices. Inside a plateau (5 of every 6 Z
   slices) this term is zero for free — so the loss isn't actually
   trying to smooth across the plateaus, just trying to align the
   discontinuities with the label's own discontinuities.

2. **Validation metric masking** — historically, validation metrics
   were computed only at the "original-label positions" (`Z % z_scale
   == 0`). This avoided counting each label voxel `z_scale` times. The
   factory still has this knob — `src/train/factory.py:372-379`
   (`valid_label_stride = z_scale`). In all-data training (no
   validation) it doesn't fire.

3. **Inference and the expert-comparison step** — the expert-annotated
   labels in `ds_test_labeled` are at *native* Z (e.g. Z=6 for a crop
   that the microscope produced as 6 slices). The model predicts at
   the upsampled Z (e.g. Z=36 after `scale_factor=6` at inference). The
   expert comparison code subsamples the prediction back to the
   label's grid before scoring; see `src/infer/expert_comparison.py:_subsample_pred_to_label_z`.

## Key code paths

- `src/utils/misc.py:_z_interpolate_and_stack` — the function that does
  it. Trilinear on channels 0-2, `np.repeat` on channel 3 (label).
- `src/utils/misc.py:resize_and_copy` — the per-TIFF driver. Now
  parallelised across a pool of CPU workers (commit `8b1d034`).
- `configs/template.yaml: experiments.default_z_scale` — the upsampling
  factor (default 6).
- `src/train/factory.py:372-379` — the validation-mask aware of the
  stacking pattern.
- `src/infer/expert_comparison.py:_subsample_pred_to_label_z` — the
  inverse, at expert-comparison time.
