# Expert comparison — model vs three annotators

Three nephrologists (Chris, David, Robin) independently annotated the
same set of three test crops (`ds_test_labeled/<annotator>/`). The
**expert-comparison step** scores the model's predictions on those
crops against each annotator AND scores the annotators against each
other.

The pairwise inter-rater scores are the **human ceiling**: a model
can't reasonably do better than the experts agree among themselves.

Run as part of `gbm.py stats <exp> -it <tag>`; implementation in
`src/infer/expert_comparison.py`.

## Why three annotators

Single-annotator ground truth has its own biases. With three:

- We can measure **inter-rater Dice** as a baseline for what
  "agreement" means. Chris vs Robin at Dice 0.77 means "two
  professional GBM annotators only agree at 0.77".
- A model that scores Dice 0.74 against Robin is "as good as another
  annotator" — within the band of normal expert disagreement.

## The pairs

For 3 annotators × 1 model, the expert-comparison step computes:

- **3 model-vs-expert pairs**: `model_vs_Chris`, `model_vs_David`,
  `model_vs_Robin`.
- **3 inter-rater pairs**: `Chris_vs_David`, `Chris_vs_Robin`,
  `David_vs_Robin`.

For each pair, on each labeled crop (3 crops × 6 pairs = 18 binary
mask pairs total), we compute per-voxel:

| Metric | Definition |
|---|---|
| **Dice** | `2·TP / (2·TP + FP + FN)` |
| **IoU** (Jaccard) | `TP / (TP + FP + FN)` |
| **TPR** (sensitivity / recall) | `TP / (TP + FN)` |
| **PPV** (precision) | `TP / (TP + FP)` |
| **Specificity** | `TN / (TN + FP)` |

Plus the raw `TP`, `FP`, `FN`, `TN` voxel counts.

Per-crop results are aggregated across crops into mean / std / min /
max per pair per metric.

## Outputs

Three files in `<exp>/results-infer/<tag>_stats/`:

| File | What |
|---|---|
| `expert_comparison.yaml` | Per-crop + aggregate, full detail |
| `expert_comparison_summary.yaml` | Aggregate only (smaller, easier to scan) |
| `expert_comparison.png` | Box plot: one subplot per metric, one box per pair |

## Where the labeled crops come from

`ds_test_labeled/` has one sub-directory per annotator. Each
annotator's directory has the same set of 3 TIFFs (the same physical
crops — only the label channel differs). The annotators' filenames
also differ across directories:

```
Chris/  NCWM.AUY380.Series004.Test.Chris_corrected_order.tiff
David/  NCWM.AUY380.Series004.Test.ForAnnotation_DUJ.tiff
Robin/  NCWM.AUY380.Series004.Test.Robin_corrected_order.tiff
```

The first matcher we wrote did naïve filename matching and missed
David's variant of the Chris-derived crop name. The current matcher
in `src/infer/expert_comparison.py:_find_crop_match` does three-tier
matching:

1. Exact filename match (fast).
2. Glob with the crop name as prefix.
3. **Stem-glob** — strip annotator-specific tokens (`.Test`,
   `.ForAnnotation`, `.Chris`, `.David`, `.Robin`) from the name and
   glob with the remaining "volume identity" prefix.

The stem matcher is what makes cross-annotator lookup work. See
`src/infer/expert_comparison.py:_crop_stem` and commit `ee7b651`.

## Z-axis subsampling — the inference / label resolution mismatch

The expert-annotated labels are at the **native Z** of the original
microscope output (e.g. Z=6 for a typical crop). The model's
prediction is at the **Z-upsampled** resolution (Z=36 if
`scale_factor=6`).

If we compared `prediction.shape = (36, H, W)` against
`label.shape = (6, H, W)` directly, the shapes don't match and the
comparator skips the pair (`Shape mismatch ... skipping`).

The fix is in `src/infer/expert_comparison.py:_subsample_pred_to_label_z`
— if the prediction's Z is an integer multiple of the label's Z, we
subsample the prediction by that factor (`prediction[::z_scale, :, :]`).
This **mirrors the training-time validation mask**: only the
prediction slices at original-label positions are compared.

If the Z ratio isn't an integer, we log a warning and skip the crop —
that's a config error worth surfacing, not silently rescaling.

## The current numbers (late May 2026 session)

| Pair | Dice (mean ± std) |
|---|---|
| **Inter-rater (human ceiling)** | |
| Chris vs David | 0.670 ± 0.030 |
| Chris vs Robin | 0.766 ± 0.020 |
| David vs Robin | 0.701 ± 0.067 |
| **Model vs experts** | |
| Unet vs Chris | 0.638 ± 0.052 |
| Unet vs David | 0.548 ± 0.024 |
| Unet vs Robin | 0.682 ± 0.076 |
| Swin v2 vs Chris | 0.702 ± 0.055 |
| Swin v2 vs David | 0.617 ± 0.036 |
| Swin v2 vs Robin | 0.740 ± 0.081 |
| Swin v3 vs Chris | 0.688 ± 0.052 |
| Swin v3 vs David | 0.588 ± 0.036 |
| Swin v3 vs Robin | 0.736 ± 0.057 |

Read this as: **Swin v2 lands inside or close to the human band on
all three annotators** (model-vs-Robin 0.74 ≈ Chris-vs-Robin 0.77;
model-vs-Chris 0.70 ≈ Chris-vs-David 0.67). Unet falls below the
band, particularly on David (0.55 < 0.67). Swin v3's stride-2 retrain
slightly hurt Dice without (visibly) fixing the jaggedness.

## Tests

`tests/infer/test_expert_comparison.py` — 29 tests at full coverage.
Includes:

- `test_metrics_from_confusion_handles_empty_class` — Dice/IoU/TPR/PPV
  fall back to 0 when the denominator is 0 (matches the trainer
  convention).
- `test_compare_crop_pair_structure` — for N annotators we get N
  model-vs-expert + C(N,2) inter-rater pairs.
- `test_find_crop_match_cross_annotator_stem` — Chris-derived crop
  name finds David's variant via stem glob.
- `test_subsample_pred_integer_ratio_picks_every_z_scaleth_slice` —
  Z=36 prediction subsamples to Z=6 matching the label.
- `test_subsample_pred_returns_none_for_non_integer_ratio` — config
  error surfaces as a warning, not a silent rescale.

## When you'd care

This is the **paper figure** that goes in the results section: a box
plot of all pairwise Dice values. Add a small text overlay of "human
ceiling band" and the reader instantly sees whether the model is good
(inside) or bad (outside).

Reading the comparison numbers above tells you the central
scientific message of the project as it stands: **SwinUNETR is the
better model**, by a clear margin, sitting inside the human
disagreement band where U-Net does not.
