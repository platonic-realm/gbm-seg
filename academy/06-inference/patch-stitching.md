# Patch stitching with Gaussian accumulation

The model trained on `[12, 256, 256]` patches but the test volumes are
much larger (typically `[78, 2164, 2164]` after Z-upsampling). At
inference time we tile the volume into overlapping patches, predict
each, and stitch the predictions back together. This doc explains how.

## The overlap trick

Naïve tiling — non-overlapping patches — produces visible seams at
patch boundaries because the model's output is less reliable near the
patch edges (the receptive field there is partially outside the
patch). We use **overlapping** patches and **Gaussian-weighted
averaging** to smooth across boundaries.

Default inference settings (for SwinUNETR with `[12, 256, 256]`
training patches):

```yaml
inference_ds:
  sample_dimension: [12, 256, 256]   # matches training (mandatory)
  pixel_stride: [3, 128, 128]        # 75% Z overlap, 50% XY overlap
  scale_factor: 6                    # Z upsampling at inference
```

With patch `[12, 256, 256]` and stride `[3, 128, 128]`, each output
voxel is hit by up to **`12/3 = 4 patches in Z`** and **`256/128 = 2
patches in each of X and Y`** — so up to `4 × 2 × 2 = 16 patches per
voxel**. The model produces 16 different predictions, each at a
slightly different patch offset; we average them.

## Why Gaussian weighting

Plain averaging treats every patch's prediction equally. But the
prediction at a voxel near the patch edge is less reliable than one
at the patch centre. The fix: weight each patch's contribution by a
**3D Gaussian centred at the patch centre**.

The Gaussian's `sigma` is set so the weight falls off smoothly across
the patch; in practice the centre voxels of the patch dominate the
output and the edge voxels just polish the seams.

Configurable via `inference.stitching` (default: `gaussian`). Other
modes: `none` (plain averaging) and `linear`.

## Where it happens — the inferer accumulator

Look at `src/infer/inference.py`. The inferer:

1. Allocates a `result_tensor` of shape `(num_classes, Z, H, W)` on
   the CPU. This holds **summed logits**, not argmax masks — averaging
   probabilities is safer than majority-voting masks.
2. Allocates a `weight_tensor` of shape `(Z, H, W)` to track the total
   Gaussian weight at each output voxel.
3. For each batch of patches:
   - Runs the model to get per-patch logits.
   - Scatter-adds `weights × logits` into `result_tensor` at the patch's
     offset.
   - Scatter-adds `weights` into `weight_tensor` at the same offset.
4. After the last batch, divides `result_tensor / weight_tensor`
   (per-voxel) and takes argmax across the class axis to get the
   final binary mask.

The accumulation runs on CPU, which is intentional: a 5 GB float32
tensor doesn't fit comfortably on the GPU alongside the model and
intermediate activations. CPU is slow for the scatter-add but only
has to do it once per patch.

## Patch enumeration — the per-volume array

`gbm.py infer` enumerates patches per volume and dispatches them.
But importantly, the SLURM-side parallelism is **per-volume** (one
array task per test volume), not per-patch. That's because each test
volume's `result_tensor` is large and needs to stay coherent within
one process; you can't split patches across processes without
collecting them again.

Inside a single inference job, the GPU processes patches in batches
of `inference_ds.batch_size` (default 8). With patch `[12, 256, 256]`,
batch=8 fits comfortably on a single 40 GB A100.

## Trade-offs

Smaller stride → more overlap → smoother stitching, more compute.
Larger stride → less overlap → faster but visible seams.

`[3, 128, 128]` (the current default) is a balance. The dense-Z-stride
test (stride `[1, 128, 128]`) for the v2 swin run showed ~1pp Dice
drop and modest visual smoothing — see [the v2-v3-v4
experiments](../07-case-study-z-jaggedness/v2-v3-v4-experiments.md).

## What "scale_factor=6, interpolate=true" does

Inference can optionally **Z-upsample the input volume on the fly**
(in addition to the resize done at `gbm.py create` time). For
`ds_test_unlabeled` volumes which are stored at native Z, this
applies the same 6× trilinear Z-upsample as training expects.

`-in true` is the typical setting; `-in false` skips the on-the-fly
upsample (used when the data is already at the trained Z resolution).
See `src/data/ds_infer.py:43`.

## The inference patch must match training

The model is built for a specific input shape, and the shape gets
baked in two places:

1. **U-Net3D's `LayerNorm.normalized_shape`** — fixed at training
   time. Inference at a different `(C, Z, H, W)` shape raises
   `RuntimeError: Given normalized_shape=...`.

2. **SwinUNETR's window_partition guard** — `window_z` is set at
   build time to match the training patch Z. Inference at a different
   Z raises `ValueError: Window Z (12) must equal input Z (24); ...`.

See [the inference-patch-must-match-train constraint](../07-case-study-z-jaggedness/diagnosis.md)
and the memory entry `reference_inference_patch_must_match_train.md`.

The practical rule: **always infer at the same `sample_dimension` you
trained with**.

## Tests

`tests/infer/test_inference_accumulator.py` (if present) and the
broader inference smoke tests cover the patch dispatching. Coverage
on the GPU path isn't huge — the trainer loop and the full dataset
class are uncovered because they'd need GPU + dataset fixtures the
CPU CI can't provide.
