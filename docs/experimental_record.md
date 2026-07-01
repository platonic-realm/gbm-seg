# GBM-seg — Experimental Record (paper source-of-truth)

**Purpose.** A self-contained record of the experiments, findings, methodology,
and code changes behind the GBM-segmentation paper. Written so that a future
author — working from this repo alone, possibly on a different machine — can
reconstruct what was done, why, and where the artifacts live. This is NOT the
paper; it is the notes the paper is written from. Keep it updated as
experiments land.

Last updated: 2026-07-01. Repo: `platonic-realm/gbm-segmentation`.
Numbers below were measured on the `main` branch at the commits listed in
§7; re-derive from the cited artifacts if in doubt.

---

## 1. Task & dataset

- **Task:** 3D semantic segmentation of the Glomerular Basement Membrane (GBM)
  from high-resolution confocal microscopy, then downstream 3D reconstruction +
  morphometry (thickness / bumpiness).
- **Input:** 3-channel TIFF z-stacks — **nephrin, collagen-4, WGA** (that
  channel order; see `src/data/ds_base.py`). On-disk axis order `ZCYX`
  (ImageJ), read as `(D, C, H, W)`. Training TIFFs carry a 4th channel = label.
- **Output:** binary GBM mask per voxel.
- **Anisotropy:** confocal Z is coarser than XY. Voxels are resampled to
  `default_voxel_size = [0.050, 0.050, 0.300] µm` at experiment-create time,
  then **Z is upsampled ×6** (`default_z_scale = 6`). Image channels are
  trilinear-upsampled; **the label channel is `np.repeat`'d** (each manual
  annotation slice is stacked 6×). This creates a period-6 structure in the
  label volume — important for interpreting the continuity results (§4, §5).
- **Statistical unit = the MOUSE, not the image.** Multiple `…SeriesNNN.tiff`
  per animal are pseudo-replicates. Disease group is derived from the animal
  token (`AUY###` / `BDP###` / `CKM###`) via `MOUSE_GROUPS` in
  `src/infer/stats.py`. Cite per-mouse (`by_mouse/`) numbers, not per-image.

---

## 2. Headline findings (what the paper reports)

### F1 — Architecture dominates; loss choice does not affect Dice
5-fold **subject-wise** CV (mouse-level splits), CrossEntropy vs Cont loss,
both architectures, 5 epochs. Metric = best-epoch validation Dice (foreground
/ GBM class), masked to real-label Z slices.

| Arch | Cont (mean ± sd) | CrossEntropy | Paired Δ (Cont−CE) |
|------|------------------|--------------|--------------------|
| **SwinUNETR** | 0.688 ± 0.054 | 0.693 ± 0.051 | −0.005 ± 0.012 |
| **3D U-Net**  | 0.604 ± 0.051 | 0.598 ± 0.069 | +0.006 ± 0.020 |

Per-fold Dice — swin cont `[0.750, 0.723, 0.687, 0.670, 0.610]`, swin CE
`[0.757, 0.733, 0.682, 0.663, 0.632]`, unet cont
`[0.657, 0.646, 0.610, 0.575, 0.534]`, unet CE
`[0.666, 0.655, 0.616, 0.541, 0.513]`.

- **Loss has no Dice effect:** the paired difference **flips sign between
  architectures** and is ~3–4× smaller than its own std and ~10× below the
  fold-to-fold spread. Cont ≈ CE on Dice, on both architectures.
- **Architecture is the real signal:** SwinUNETR 0.69 vs 3D U-Net 0.60
  (Δ≈0.09), ~15× any loss effect, far outside the noise. **Report the two
  architectures side by side; SwinUNETR is clearly better here.**

### F2 — The continuity (Cont) loss DOES work: it makes the reconstruction measurably smoother along Z (an effect Dice cannot see)
Final **all-data** models (trained on every subject, no holdout, 8 epochs),
SwinUNETR, Cont vs CE. Inference on the held-out **test set** (n = 62
whole-glomerulus volumes). Z-continuity metrics on the raw predictions
(PSP-processed values are near-identical):

| Metric (raw) | Cont | CE | Δ | ~z |
|--------------|------|----|---|----|
| **Z-TV per fg-voxel** ↓ | 0.1342 ± 0.0186 | 0.1447 ± 0.0187 | **−7.2%** | −3.1 |
| **Flips per fg-column** ↓ | 2.415 ± 0.223 | 2.605 ± 0.229 | **−7.3%** | −4.7 |
| **Adjacent-slice IoU** ↑ | 0.8841 ± 0.0129 | 0.8759 ± 0.0130 | **+0.9%** | +3.5 |
| Foreground fraction | 0.0600 | 0.0604 | ≈ equal | — |

- **All three independent metrics agree**, same direction, |z| ≈ 3–5 over 62
  volumes → the ~7 % smoothness gain is well outside the noise.
- **Fairness control holds:** foreground fraction is equal (0.060 vs 0.060), so
  Cont is NOT "smoother" merely by predicting less membrane — it predicts the
  same amount, more Z-coherently. This is the key confound and it is ruled out.
- **Interpretation:** CE columns cross the membrane 2.61× on average, Cont
  2.42× — closer to the ideal 2 (clean enter+exit), i.e. fewer spurious
  Z-flicker crossings.
- **Why Dice missed it:** validation Dice is masked to the *real-label* Z slices
  only (`_select_real_z`), while the continuity term acts on all Z — its whole
  job is the *interpolated in-between* slices Dice never scores. So by
  construction Dice can only see Cont's small CE-down-weighting cost, never its
  smoothness upside. The dedicated continuity analysis (§5) was built to
  surface exactly this.

**Verdict for the paper:** the continuity loss costs nothing on segmentation
accuracy and yields a modest but statistically reliable improvement in Z-axis
reconstruction smoothness. It is a free-or-better choice for the final model
where 3D reconstruction quality / morphometry matters.

### F3 — Roughness of the reconstruction is a training-DURATION effect
Independent earlier investigation (SwinUNETR): identical model/aug, early
snapshot (~3 k steps) gives smooth Z output; long training (~42 k steps)
overfits high-frequency detail → jagged Z output. Both aug and no-aug arms show
the same pattern. The final runs deliberately stop at ≤16 k steps (8 epochs) to
stay well below this regime — relevant because F2 studies smoothness.
Note: CV `best_epoch` was 5/5 (still rising at the 5-epoch cap) → 5 epochs
under-trains; all-data finals used **8 epochs (~16 k steps)** as a measured
extension that stays below the roughening regime.

### F4 — Offline augmentation: Dice-neutral but morphometry-moving
Earlier SwinUNETR ablation (offline aug on vs off; see archived prior runs).
Held-out expert comparison: Dice tied (aug 0.696 / noaug 0.704, both ≈ the
~0.71 inter-annotator ceiling). Recall/precision trade: aug → higher recall
(TPR ≈ 0.80), noaug → higher precision (PPV ≈ 0.67). **Downstream GBM thickness
differs ≈46 %** (aug Collagen 452 nm vs noaug 309 nm) — the aug choice barely
moves Dice but strongly changes the morphometric output (the actual scientific
quantity). Watch this when choosing the final model. Caveat: small n (3 crops,
2 Collagen mice).

---

## 3. Final method configuration (Materials & Methods)

Master template `configs/template.yaml` (each experiment gets a frozen copy at
`gbm.py create` time). Values as of the commits in §7:

- **Preprocessing:** resample to `[0.050, 0.050, 0.300] µm/vox`; Z-upsample ×6
  (images trilinear, labels `np.repeat`).
- **Patch geometry:** train/valid/infer sample dimension `[12, 256, 256]`
  (Z, X, Y) — MUST match across train and inference (LayerNorm / window-attn
  bake in the patch shape). Train stride `[8, 192, 192]`; inference stride
  `[1, 64, 64]` (dense overlap, Gaussian stitching).
- **Models** (both reported):
  - **SwinUNETR** (custom, NOT MONAI): `feature_size=24`,
    `depths=[2,2,2,2]`, `num_heads=[3,6,12,24]`, `window_size_xy=7`,
    `patch_embed_z_kernel=5` (3-slice Z-conv stem, partially smooths the
    period-6 stacked-label pattern before attention), `z_deduction=auto`,
    gradient checkpointing on (needed at batch ≥10 on 80 GB).
  - **3D U-Net:** `feature_maps=[64,128,256,512]`, `z_deduction=auto`,
    gradient checkpointing `auto` (→ off on ≥60 GiB GPUs; conv net doesn't
    need it there).
- **Loss (ablated axis):**
  - **Cont** = `0.7·CE + 0.3·continuity`, CE class weights `[3.0, 7.0]`
    (bg, GBM). Continuity term = mean `|softmax_probs[z+1] − softmax_probs[z]|`
    over the class dim, diffed along depth (`src/train/losses/loss_cont.py:
    continuity_loss_diff`). Fixed weights (a prior learned-weight mechanism was
    replaced).
  - **CrossEntropy** = plain weighted CE, weights `[3.0, 7.0]`.
  - (Dice, IoU exist and are commented out in `ablation_specs/loss_*.yaml`;
    the focused study is CE vs Cont only.)
- **Optimizer / schedule:** Adam, lr `1e-4` (×√world_size under DDP);
  `poly_decay` power 0.9, per-epoch step; warmup 0.
- **Epochs:** 5 for CV, **8 for the all-data finals** (justified by F3).
- **Augmentation (fixed across the loss ablation):** offline
  `[_zoom 0.7, _zoom 1.3, _twist_clock 0.5]` (precomputed to disk, hardlinked
  across cells), online on (rotate/blur/crop/channel-drop). NB: the `_twist`
  offline recipe + online-on was the settled choice; earlier runs also explored
  zoom-1.5 and online-off.
- **Batch:** **10 per rank** for the finals. NB: batch **12** on 8×A100-80 GB
  trained fine but hit a GPU OOM in the end-of-fold NCCL all-reduce (97 % memory,
  no teardown headroom); **10 is the safe value** (~67 GB, ~13 GB headroom).
- **DataLoader:** train workers 3, valid workers 1, `persistent_workers=True`.
  Host-RAM sizing matters (see §6 memory fix); at batch 10 the job peaks
  ~626 GB of the 900 GB cgroup on the 8-GPU node.
- **Distribution:** DDP (torchrun), 8 GPUs/run, mixed precision. Validation set
  is sharded across ranks via `DistributedSampler(drop_last=False)`.
- **Seed:** 88233474 (python/numpy/torch/cuda + per-worker).

**Hardware used:** source cluster `lyn` (8×A100-80 GB node `lyn-gpu-06`);
migrated mid-project to `ramses` (H100 nodes, `gpu` partition, 4×H100-80 GB/node,
QOS cap 8 GPUs/user). See §8.

---

## 4. Statistical methodology

- **Cross-validation:** 5-fold, **subject-wise** (splits at the mouse level;
  `fold_assignments.yaml` copied identically into every ablation cell, so
  cont/CE folds use the SAME splits → per-fold differences are paired).
- **Model selection within a fold:** best validation Dice across epochs
  (early-stopping-style). This is mildly optimistic (peeks at the val fold) but
  applied identically to every cell, so the comparison is fair.
- **Loss comparison (F1):** paired per-fold Δ (same splits). Report mean ± sd of
  the paired difference; with n=5 folds the pairing matters. Direction-flip
  across architectures is the evidence of "no effect."
- **Continuity comparison (F2):** per-volume metrics over the test set (n=62),
  compared with an approximate Welch z (Δ / pooled SE). Report the fairness
  control (equal foreground fraction) alongside.
- **Metric-computation caveats (state in Methods; all consistent across cells,
  so they don't bias the comparisons):**
  1. Validation metrics are **average-of-batch-averages**, not a voxel-
     population mean — for nonlinear Dice these differ slightly from a dataset-
     level Dice.
  2. `DistributedSampler(drop_last=False)` pads the validation shard to be
     divisible by world size by repeating samples → a few validation patches
     are double-counted.
  3. Validation Dice is **masked to real-label Z slices** (`_select_real_z`) —
     interpolated slices are not scored (this is why Dice can't see F2).
- **Morphometry stats** (`src/infer/stats.py`): thickness treated as
  **lognormal**; per-sample IQR outlier fences on the **log** scale; PSF
  left-censored MLE fit on the pre-IQR data. Two group analyses written:
  `by_mouse/` (unit = mouse, **the one to cite**) and `publication/by_image/`
  (unit = image, transparency only, overstates n). Figure set per group:
  SuperPlot, estimation plot (Cumming/Gardner-Altman), group ECDF, raincloud,
  effect-size forest (Cliff's δ + CI), plus `group_significance.yaml`
  (Kruskal-Wallis + pairwise Mann-Whitney / Bonferroni). With 2–4 mice/group
  the tests are underpowered post-correction → foreground effect size + CI.

---

## 5. The continuity analysis (a methodological contribution)

New in this work: a dedicated, dataset-agnostic step quantifying **along-Z
smoothness** of a prediction — the axis the continuity loss targets but Dice
cannot measure. Code: `src/infer/continuity.py`. CLI: `gbm.py continuity
<exp> -it <tag>` → `results-infer/<tag>_continuity/continuity_result.yaml`;
`gbm.py continuity-compare EXP:TAG …` → side-by-side table. sbatch:
`sbatch/continuity.sbatch`. Scores BOTH raw `prediction.npz` and PSP
`prediction_psp.npz`, per sample, then aggregates (mean/sd/min/max/n).

**Metric definitions** (on a binary `(Z,H,W)` mask; `fg` = foreground voxel
count):
- **Z-TV per fg-voxel** (headline): `Σ|mask[z+1]−mask[z]| / fg` — number of
  along-Z boundary crossings normalised by membrane volume. Lower = smoother.
  The `/fg` normalisation is the fairness control (a model predicting less
  foreground has trivially fewer crossings).
- **Mean flips per fg-column:** for each `(x,y)` column with any foreground,
  count 0↔1 transitions along Z; average over fg columns. A clean single
  membrane crossing = 2; jagged > 2.
- **Mean adjacent-slice IoU:** IoU(`mask[z]`, `mask[z+1]`) averaged over slice
  pairs where either is non-empty. Higher = smoother.
- Also reported: foreground voxel count + fraction, Z depth.

Degenerate volumes (single Z slice or empty) yield NaN metrics (excluded from
aggregation), not a crash. Tests: `tests/infer/test_continuity.py` (smooth <
jagged ordering, exact flip counts, volume-fairness of the normalisation,
degenerate handling, end-to-end, compare table).

---

## 6. Code changes in this campaign (reproducibility + correctness)

Several were **correctness fixes that could have corrupted results** if left —
call these out in the paper's reproducibility statement. Commit SHAs in §7.

**Result-affecting bug fixes:**
- **Ablation cell re-rooting** (`aa1599e`): the ablation runner copied the base
  config verbatim, so every cell's `root_path` pointed at the BASE experiment.
  All folds wrote `results-train/` into the base and sibling cells (cont vs CE)
  collided there, overwriting each other. Fixed to re-root each cell at its own
  dir. **All F1/F2 numbers are from runs after this fix**; cont/CE separation
  was verified at the file level (distinct inodes, distinct per-fold values,
  base dir empty).
- **CV aggregation** (`799998e`): `_aggregate_cv` discovered metric names from
  `per_fold[0]` only, so if fold 0 died the aggregate was silently empty even
  when other folds succeeded. Fixed to use the union of all folds.
- **Metric formula fixes** (`a820a94`): `PhiCoefficient` (MCC) rewritten to
  `(TP·TN − FP·FN)/√(…)` with float casts (was sum-based, int64-overflowing);
  `FowlkesMallowsIndex` = `√(PPV·TPR)` (was `√(PPV+TPR)`).
- **Running-metric robustness** (`a7c0ff1`): `GPURunningMetrics.add` coerces
  scalars to tensors and skips non-finite (nan AND inf) batches.

**Config / infra changes:**
- **gradient_checkpointing default `auto`** (`da0aae1`): was hardcoded `on`,
  which made the conv U-Net checkpoint needlessly on 80 GB cards (~30 % slower,
  no memory benefit). `auto` = off on ≥60 GiB. Applies to BOTH models (not
  Swin-only as an old comment claimed).
- **Per-worker host-RAM cut** (`325842e`): `ds_train.__getitem__` kept the label
  channel as a view and deferred binarise+int64 cast to the CROPPED patch (was
  a ~2.7 GB/call full-volume int64 copy that glibc retained). `train_ddp.sbatch`
  sets `MALLOC_ARENA_MAX=2` + `MALLOC_TRIM_THRESHOLD_` so glibc returns freed
  volumes to the OS. Together these dropped per-worker RSS from ~22–30 GB to
  ~13 GB and are why batch-10 / 3-workers fits the 900 GB cgroup.
- **W&B provenance** (`74b87f6`, `5650b67`, `088437d`): run names + tags carry
  the **cluster** (`lyn`/`ramses`) and the **actual fold** (fixed a bug where
  the ablation runner's static `__fold0` cell name mislabelled every fold as
  fold-0); per-cluster W&B project (`gbm-ablation-lyn` vs `-ramses`), applied to
  BOTH the per-fold and the cv_summary runs.
- **DDP GPU flexibility** (`31accd4`): torchrun `nproc` derives from
  `SLURM_GPUS_ON_NODE`, so GPU count is controlled purely by the sbatch
  `--gres` flag.
- **Continuity feature** (`25ffc5f`): see §5.
- **Stats overhaul** (`1ac9808`): curated group-comparison figures, with_mask /
  without_mask variants, per-mouse analysis, single configurable
  `max_thickness` knob (= the clipping threshold). See `docs/stats_pipeline.md`.

Tests: full suite ~358 tests, CPU-only (`python -m pytest tests/`); lint
`ruff check .`.

---

## 7. Provenance — commits (this campaign, newest first)

```
25ffc5f feat(continuity): Z-jaggedness analysis for inference (cont-vs-CE)
799998e fix(cv): aggregate metrics from union of folds, not per_fold[0]
aa1599e fix(ablation): re-root each cell at its own dir (root_path)
088437d fix(wandb): cv_summary run also uses per-cluster project + prefix
655a8b5 config(ablation): bump epochs 2 -> 5
da0aae1 fix(checkpointing): default gradient_checkpointing to 'auto', not 'on'
2aed579 config(ablation): scope to CrossEntropy vs Cont only (comment out Dice/IoU)
f32da00 config(ablation): validated lg6 base config + one-shot launcher
325842e perf(data)+wandb: cut per-worker host RAM; per-cluster W&B project
5650b67 fix(wandb): derive cluster from node name when SLURM_CLUSTER_NAME is (null)
74b87f6 feat(wandb): cluster prefix in run name/tags + fix per-fold mislabel
d282ed5 config(ablation): finalize loss-ablation base config
31accd4 feat(ddp): derive torchrun nproc from SLURM_GPUS_ON_NODE
a7c0ff1 fix(metrics): GPURunningMetrics robust to scalar + non-finite
a820a94 fix(metrics): correct PhiCoefficient (MCC) and FowlkesMallowsIndex
1ac9808 feat(stats): group-comparison figure overhaul, mask variants, per-mouse
```

---

## 8. Where the artifacts live (to regenerate figures/tables)

Experiments root: `/projects/ag-bozek/afatehi/gbm/experiments/` (cluster path;
authoritative). Prior experiments archived to `…/gbm/experiments.2026/`.

**Loss ablation (F1)** — 5-fold CV cells, each with `cv_results.yaml`
(per-fold + aggregate) and `results-train/fold_N/best_metrics.yaml`:
- `lossabl_swin__cont__fold0`, `lossabl_swin__crossentropy__fold0`
- `lossabl_unet__cont__fold0`, `lossabl_unet__crossentropy__fold0`
  (the `__fold0` suffix is a naming artifact; each cell holds all 5 folds.)

**All-data finals + continuity (F2)** — reuse the swin cell dirs; snapshots
under `results-train/snapshots/all_data/008-16000.pt`:
- Cont: inference tag `alldata8ep_cont` →
  `results-infer/alldata8ep_cont_continuity/continuity_result.yaml`
- CE: inference tag `alldata8ep_ce` → `…alldata8ep_ce_continuity/…`
- Test set = `datasets/ds_test_unlabeled/` (n=62 whole-glom volumes).

**Reproduce a comparison:**
```
python gbm.py continuity-compare \
  lossabl_swin__cont__fold0:alldata8ep_cont \
  lossabl_swin__crossentropy__fold0:alldata8ep_ce
```

W&B: project `gbm-ablation-lyn` (this campaign ran on `lyn`), entity
`gbm-project`. Runs named `lyn-<cell>-fold-N-eager` + `…-cv-summary`.

Snapshots/predictions are large and NOT in git; only code + configs +
`cv_results.yaml`-scale outputs are. Raw dataset lives outside the repo at the
cluster `default_data_path` and is copied in by `gbm.py create`.

---

## 9. Open items / next steps for the paper

- **Expert / test-set evaluation (Stage C):** score the final models on
  `ds_test_labeled` (3 annotators → STAPLE consensus + inter-rater) — the
  "model lands inside the inter-rater envelope" headline. Test set touched ONCE,
  on final models only (hygiene: it's only 3 small crops).
- **Whole-glom morphometry (Stage D):** run the final model(s) on
  `ds_test_unlabeled`; thickness/bumpiness stats per mouse. Tie F2's smoothness
  gain to a morphometric quantity (does smoother Z change thickness estimates?).
- **Both architectures reported** (SwinUNETR + 3D U-Net) — different
  compute/memory footprints are themselves a finding.
- **Unet continuity:** F2 is SwinUNETR only; run the same continuity comparison
  on the U-Net finals if the loss story needs both architectures.
- **Visualise F2:** side-by-side Z-slice / 3D renders of cont vs CE to *show*
  the smoothness difference, not just tabulate it.
- Decide the final model + loss: SwinUNETR is the accuracy winner; Cont vs CE is
  a free choice on Dice, with Cont giving smoother reconstructions (F2).
