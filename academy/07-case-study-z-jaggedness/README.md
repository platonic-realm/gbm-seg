# Case study — the Z-jaggedness saga

This case study is the most useful single thing in the Academy. It
pulls in concepts from [data](../02-data/z-anisotropy-and-upsampling.md),
[models](../03-models/swinunetr.md), [loss](../04-loss-cont.md),
[training cadence](../05-training/samples-based-reporting.md),
[inference patches](../06-inference/patch-stitching.md), and
[expert comparison](../06-inference/expert-comparison.md) to tell one
coherent story about how a real research problem unfolded over a week.

## The problem in one sentence

When you visualise SwinUNETR's segmentation output as a 3D rendered
TIFF or MP4 and step through the Z slider, the predicted GBM mask
**transitions in 6-slice stair-steps along Z**, whereas the 3D U-Net's
output is smooth.

It's not a quantitative failure — SwinUNETR's Dice against expert
annotators is **higher** than the U-Net's. It's a *qualitative*
failure that shows up only in the renders: the GBM should look like
a continuous, curved sheet, and it looks like a staircase instead.

## Why it matters

Three downstream consequences:

1. **Morphometry noise.** [Thickness measurement](../06-inference/morphometry.md)
   is based on the 3D distance transform inside the predicted mask. A
   staircased mask gives a staircased thickness map → noisier
   thickness histograms → less precise statistics for the paper.
2. **Visual credibility.** A reviewer looking at the rendered MP4
   immediately notices the staircase. It undermines confidence in the
   model regardless of what the Dice numbers say.
3. **It's the symptom of a deeper model behaviour** — namely that
   SwinUNETR is **memorising** an artificial periodicity in the
   training target rather than learning the underlying biological
   structure. That's worth understanding and fixing.

## Files in this section

1. **[Diagnosis](diagnosis.md)** — the mechanism: why SwinUNETR has
   the capacity to memorise the 6-slice stacked-label period and why
   U-Net doesn't. The interplay of [Z label
   upsampling](../02-data/z-anisotropy-and-upsampling.md),
   [window attention with relative position
   bias](../03-models/window-attention.md), and [conv inductive
   bias](../03-models/inductive-biases.md).

2. **[Experiments v2 → v3 → v4](v2-v3-v4-experiments.md)** — the
   experimental sequence we tried. Each variant tested one hypothesis.
   What worked, what didn't, what we learned along the way.

3. **[Future directions](future-directions.md)** — what to try next
   if the current v4 ablations don't fully solve it: window-Z capping,
   soft labels, mesh-based relabelling, distance-transform labels.

## How to read this case study

If you only have time for one file, read [diagnosis](diagnosis.md) —
it has the conceptual core. The experiments are interesting if you
want to see the methodology of "isolate the hypothesis"; future
directions are interesting if you want to extend the work.

## Quick summary of the state at end of session 2026-05-30

- **v2** (`swin_offline_a100_v2`): train stride `[3, 128, 128]`,
  trained Dice peaked at ~0.74. Visibly jagged Z output.
- **v3** (`swin_offline_a100_v3`): train stride `[2, 128, 128]`
  (heavier phase coverage), trained Dice peaked at ~0.71. **Still
  visibly jagged** — proved the problem wasn't fixable by train-time
  stride alone.
- **v4** (in progress as of this writing): two parallel variants
  testing **architectural** interventions:
  - **v4_nobias**: `use_relative_pos_bias=False` — kills the
    [bias mechanism](../03-models/window-attention.md) that enables
    period-6 memorisation.
  - **v4_winxy3**: `window_size_xy=3` (was 7) — starves the
    long-range XY attention fallback that lets the model predict each
    Z slice independently.

If neither v4 ablation fixes the jaggedness, the next step is the
**window_z capping** intervention (see [future
directions](future-directions.md)) — relax the `window_partition`
assertion that `Z == win_z`, and properly partition Z into windows
smaller than `stage_z`. That's a bigger code change.
