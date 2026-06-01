# Morphometry — thickness via distance transform

After PSP cleans the predicted mask, the next stage measures **3D
thickness** at every voxel inside the GBM. This is the clinical
output of the whole pipeline: a per-voxel thickness map per glomerulus,
which can be aggregated into thickness histograms, polar plots,
top-down views, and ultimately the disease classifier.

`gbm.py morph <exp> -it <tag>`; implementation in
`src/infer/morph.py:Morph`.

## What "thickness" means in 3D

For a thin 2D surface embedded in 3D (the GBM), thickness at a point
on the surface is the **distance to the nearest opposite face**. For
a voxel mask, that's the **distance transform inside the mask** —
for each foreground voxel, compute the distance to the nearest
background voxel, then double it (because the voxel sits halfway
between two opposite faces).

Mathematically: if `D(v)` is the 3D Euclidean distance from voxel `v`
to the nearest background voxel, then `thickness(v) ≈ 2 × D(v)`.

For a uniform membrane, this gives a flat number. For a thickening
or thinning region, the map shows local variation.

## The implementation

`Morph` runs `scipy.ndimage.distance_transform_edt` on the PSP mask.
Output is a `(Z, H, W)` float32 array of thickness values in voxels.
We convert to physical units by multiplying by the **voxel pitch in
the direction of the local thickness gradient** — but in this dataset
the voxels are cubic (post Z-upsampling), so it's just `2 × D(v) ×
voxel_size_um`.

Writes `distance_result.npz` per volume.

## PSF correction

Confocal microscopes don't actually image at the perfectly small
voxel size we Z-upsampled to. The real spatial resolution along Z is
worse — the **point spread function** (PSF) blurs the signal axially
across multiple Z slices.

If the GBM is 300 nm thick (1 voxel at the upsampled 0.05 µm Z pitch),
but the Z PSF has a 600 nm FWHM, the measured "thickness" will be
inflated by the PSF blur. The morph step optionally applies a
**PSF correction**: subtracts an empirically calibrated PSF kernel
size from the raw thickness, clipped at zero.

Writes a PSF-corrected variant alongside the raw `distance_result.npz`
(`psf_result.npz`). Both go to downstream `stats`.

## Parallelism — per-volume array

Each volume is independent. The morph parent (`sbatch/morph.sbatch`)
submits an array — one task per volume — each running on its own GPU
(some intermediate steps use scipy on CPU but the distance transform
is fast on CPU anyway).

Typical wall time: ~90 minutes for 15 test volumes in parallel.

## What stats does with it

The `stats` stage reads `psf_result.npz` (or the raw `distance_result.npz`)
and produces:

- **Thickness histograms** at multiple bin counts (10, 20, 50, 100)
  per volume.
- **Cylindrical / polar plots** projecting the thickness onto a 2D
  polar coordinate system aligned with the glomerular geometry.
- **Top-down views** — 2D projections with aspect-ratio correction
  for visual inspection.
- **Aggregate** statistics across all volumes (mean, std, percentiles
  of thickness).

`configs.inference.morph.thickness_clip_max` (default 1400) caps very
high thickness values that are usually PSF artifacts, not real
biology. `gbm.py stats --clipping` enables it; without the flag the
raw distribution is reported.

## Other shape metrics — bumpiness

The "thickness map" is the primary metric, but the user also analyses
**bumpiness** — local variation in thickness as a function of position.
This is computed in stats (not in morph) and shows up in the
comparative box plots.

## Tests

`tests/infer/test_morph.py` (if present) — there's some coverage in
the broader test suite but not detailed enough to constitute a
regression for clinical metrics. The morph step is one of the
weaker-tested parts of the pipeline, alongside the trainer loop and
the full dataset class (both of which need real GPU + dataset
fixtures the CPU CI can't provide). Improving this is a clean
follow-up.

## Why this is the project's real output

If you're a paper reader, the segmentation Dice numbers in the
expert-comparison results matter for benchmarking, but the **thickness
histograms** matter for the clinical conclusion. A thickening trend
in patient cohort A vs control cohort B is what the morph stage
produces, and the comparative box plots are the figures the paper
will lean on.
