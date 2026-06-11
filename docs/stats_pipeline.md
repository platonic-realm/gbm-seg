# Statistical analysis pipeline — what it does and how to read it

This document describes the `gbm.py stats` stage end-to-end: what each
computation means biologically, what assumptions it makes, how to read
the output files, and what known biases remain. The code lives at
[`src/infer/stats.py`](../src/infer/stats.py); tests are in
[`tests/infer/test_stats.py`](../tests/infer/test_stats.py).

The pipeline produces three kinds of artifacts:

1. **Per-sample**: histograms of voxel-level GBM thickness, top-down
   thickness map, polar-coordinate average thickness, distinct
   visualisations.
2. **Cohort-level**: comparative box plot across samples and a
   sample-weighted aggregate (mean of per-sample means with SEM).
3. **Group-level**: significance tests across the Control / Podocin /
   Collagen disease groups, with effect sizes.

---

## 1. Pipeline order and inputs

`gbm.py stats <exp> -it <tag>` runs after the morph stage has written
per-sample thickness arrays. Inputs per sample, all under
`<exp>/results-infer/<tag>/<sample_name>/`:

| File | Content | Source |
|---|---|---|
| `psf_result.npz` | (Z, H, W) PSF-corrected thickness in nm, voxel-wise; 0 = no GBM here OR PSF-clamped | morph stage |
| `prediction.tif` | (Z, C+1, H, W) image channels + mask | inference / labels-as-pred |
| `psf_clamp_stats.yaml` | per-sample clamp diagnostics (count, percentage, PSF values used) | morph stage |

Outputs all go to `<exp>/results-infer/<tag>_stats/`:

- `summary_statistics.npz` — per-sample row of (q1, median, q3, ..., mean, std, censored_mean, censored_ci_*)
- `aggregated_thickness_data.npz` — every voxel from every sample concatenated, voxel-weighted
- `sample_weighted_aggregate.yaml` — cohort-level mean (raw and bias-corrected)
- `group_significance.yaml` — Kruskal-Wallis omnibus + pairwise Mann-Whitney with Bonferroni + Cliff's delta
- `comparative_box_plot.png` — IQR box plot across samples
- per-sample subdir with histograms, polar plots, top-down views, combined views
- `metadata.txt` — parameters, samples, generated files, flag annotations

---

## 2. Per-sample preprocessing

For each `psf_result.npz`:

### 2.1. Hard 1200 nm cap (always applied)

Thickness values above **1200 nm** are biologically implausible (the
pathology literature places normal GBM at ~250–400 nm and even
hypertrophic disease at < 1000 nm; > 1200 nm voxels are measurement
artefacts — typically from the distance transform running away through
incompletely-segmented capillary lumens). These voxels are masked to 0.

Constant: `HARD_THICKNESS_LIMIT_NM = 1200` in
[`src/infer/stats.py`](../src/infer/stats.py). The hard cap is applied
**unconditionally**, separately from any `--clipping` flag the user
passes. In v6 inference, this cap masks **100K–1.2M voxels per sample**
— confirms the cap is doing real work, not pruning legitimate values.

### 2.2. COL4 channel masking (uniform across groups)

The COL4 channel highlights the glomerular boundary / capillary lumens.
We use the *high* COL4 voxels as a mask of where the GBM cannot be, and
zero those voxels in the thickness map before stats.

**Previously**: the threshold was group-specific —
`Control: P91, Collagen: P97, Podocin: P92.5`. This creates a
**systematic measurement bias that mimics biological group differences**:
a sample with a stricter (higher-percentile) threshold leaves fewer
voxels in the GBM, including only the thinnest core. Comparing across
groups under unequal thresholds confounds biology with measurement
convention.

**Now**: uniform `UNIFORM_MASK_PERCENTILE = 95` across all groups. Same
measurement convention everywhere → cross-group differences reflect
biology.

### 2.3. Morphological opening (3D)

After thresholding, the binary mask is cleaned with a 3D opening
(erosion then dilation, structure = `np.ones((1, 3, 3))`). **Previously**
this was a 2D opening per Z slice, which produced inconsistent results
across the Z-upsampled stack of duplicate slices.

### 2.4. Outlier removal (proper Tukey IQR)

The cleaned voxel-level thickness array is then passed through Tukey IQR
outlier removal. Q1 / Q3 are the standard 25th / 75th percentiles (was
previously 5th / 95th, which gave a fence at ~6.6σ for normal data —
effectively a no-op). Both lower and upper fences are enforced (was
previously upper-side only).

After this pipeline, what remains is a clean voxel-level thickness
array per sample.

---

## 3. Per-sample statistics

From the cleaned voxel-level thickness data, we compute:

| Field | Formula | Notes |
|---|---|---|
| `q1` | percentile 25 | proper IQR (was P5, misleading) |
| `median` | percentile 50 | |
| `q3` | percentile 75 | proper IQR (was P95) |
| `lowerfence` | `max(min(data), Q1 - 1.5·IQR)` | Tukey lower whisker |
| `upperfence` | `min(max(data), Q3 + 1.5·IQR)` | Tukey upper whisker |
| `mean` | `np.mean(data)` | **biased upward** by PSF clamping (§4) |
| `std` | `np.std(data, ddof=1)` | sample std (was previously ddof=0 = pop std) |

Plus the bias-corrected fields — see §4.

These rows are saved to `summary_statistics.npz` (numpy structured
array, one row per sample, fields above + group label + censored_*).

---

## 4. The PSF-clamping bias and its statistical correction

This is the most subtle and the most important part of the pipeline.

### 4.1. What is PSF clamping?

The morph algorithm corrects for the imaging system's point-spread
function (PSF). For each surface voxel, it computes:

```
corrected_thickness² = measured_thickness² − (PSF_lat² · cos²α + PSF_ax² · sin²α)
```

where `α` is the angle between the GBM surface normal and the imaging
axis, `PSF_lat = 149 nm`, `PSF_ax = 434 nm`. This subtracts the imaging
system's spread in quadrature, recovering the actual thickness.

**Problem**: when `measured_thickness²` is smaller than the PSF term,
the corrected thickness would be imaginary. In that case, the algorithm
silently sets it to 0. The voxel becomes invisible to the rest of the
pipeline.

This is **left-censoring at a known threshold**. The true thickness
exists, but if it's below ~149 nm (the lateral PSF), the measurement
falls below resolution and the corrected output is set to 0 instead.

`psf_clamp_stats.yaml` per sample reports `clamp_count` (number of
surface voxels where clamping fired) and `clamp_percentage`. For the v6
Collagen cohort, this percentage runs **20–44%** on every sample. The
disease causes membrane thinning → many sub-PSF measurements → high
clamp activation.

### 4.2. Why the naive `np.mean(data[data != 0])` is biased

We exclude the clamped (0-valued) voxels from `mean` — those are the
voxels we *don't have a real measurement for*. But the data we keep is
the **upper portion** of the true thickness distribution (everything
above the ~149 nm resolution floor). Averaging only that upper portion
**overestimates** the true mean.

The bias is roughly equal to the missing-tail contribution: if 30% of
voxels are below 149 nm, the reported mean is shifted up by approximately
`E[T | T > 149nm] − E[T]`, which is positive and depends on the
distribution shape and clamping fraction.

For Collagen samples at 30% clamping, the bias can easily be 30–50 nm
upward — a meaningful fraction of the reported mean.

### 4.3. The statistical correction we apply

We treat the per-voxel thickness as a sample from a **lognormal**
distribution (biological thickness measurements are typically lognormal
— positive, right-skewed, multiplicative noise). We fit the lognormal
parameters using the censored maximum-likelihood objective:

```
ℓ(μ, σ | data) =  Σᵢ log f(xᵢ; μ, σ)                  ← uncensored voxels
                + n_clamped · log F(PSF_lat; μ, σ)    ← censored voxels
```

where `f` / `F` are the lognormal pdf / cdf. The uncensored term is
exact — these are voxels where we have a measurement and we score them
under the lognormal pdf. The censored term contributes
`n_clamped × log P(T < PSF_lat)` — each clamped voxel contributes the
log-probability that its true thickness was below threshold, evaluated
at the current `μ, σ`. The MLE jointly fits both.

The bias-corrected mean is then the mean of the recovered lognormal:

```
mean_corrected = exp(μ + σ²/2)
```

A **95% confidence interval** comes from the delta method: numerically
approximate the Hessian of the negative log-likelihood at the optimum,
invert to get the asymptotic covariance of `(μ, log σ)`, then propagate
through the gradient of the mean.

Implementation: [`fit_lognormal_left_censored()`](../src/infer/stats.py).
Optimisation uses Nelder-Mead (more robust than BFGS for this
objective). The Hessian for CI is computed by central differences.

### 4.4. What the corrected fields look like in the summary

Per-sample, the structured array `summary_statistics.npz` now carries:

```
sample_name, group,
q1, median, q3, lowerfence, upperfence,
mean,                    ← raw mean (biased high when clamping non-trivial)
std,
censored_mean,           ← bias-corrected mean from left-censored MLE
censored_median,         ← exp(μ); the lognormal median
censored_ci_lo,          ← 95% CI lower bound
censored_ci_hi,          ← 95% CI upper bound
censored_converged       ← bool: True if the fit succeeded
```

### 4.5. Why we don't just exclude high-clamp samples

The earlier approach was to *exclude* samples with high clamp from
cross-sample aggregation. But for thin pathological tissue (Collagen
here), **high clamping is the biological signal** — the disease causes
thinning. Excluding clamped samples removes exactly the data we want
to study. The current convention: include everything, flag (in
`metadata.txt`) the samples with clamp > `PSF_CLAMP_WARNING_PCT = 20%`,
and let the user read the flag alongside the corrected mean.

### 4.6. Why the rank-based group tests don't need the same correction

For *group comparisons* (Control vs. Podocin vs. Collagen), we use
Mann-Whitney U + Cliff's delta. These are **rank-based**: they only
care about the order of values, not the absolute magnitudes. If
censoring moves the same fraction of voxels to the bottom of every
sample, the relative rankings between groups are preserved. So the
**p-values and effect sizes are approximately correct even without the
MLE correction**.

What we'd cite in a paper:

- **For absolute thickness** (Control GBM = 380 nm): the **`censored_mean`**.
- **For group differences** (Collagen GBM is thinner than Control): the
  Mann-Whitney U p-value + Cliff's delta from `group_significance.yaml`.

---

## 5. Cohort-level aggregation

### 5.1. Voxel-weighted aggregate

`aggregated_thickness_data.npz` concatenates every voxel from every
sample. Larger samples contribute proportionally more voxels. Useful
for plotting overall distributions, not for biological comparisons
(which should weight each sample equally regardless of voxel count).

### 5.2. Sample-weighted aggregate

`sample_weighted_aggregate.yaml` — the appropriate biological summary.
Each sample contributes one number (its mean) to the cohort summary.
The file has two blocks:

```yaml
n_samples: 15
raw:
  mean_of_sample_means: 412.3        # ← biased high when PSF clamping non-trivial
  std_of_sample_means: 38.1
  sem: 9.8                            # SEM = std / sqrt(n)
  median_of_sample_means: 408.5
  note: "Raw mean — biased upward when PSF clamping is non-trivial."
censored:
  n_samples: 15
  mean_of_sample_means: 365.7        # ← bias-corrected via per-sample MLE
  std_of_sample_means: 41.3
  sem: 10.7
  median_of_sample_means: 360.1
  method: "Per-sample lognormal MLE with left-censoring at the lateral PSF (149 nm)."
```

The `censored.mean_of_sample_means` is the cohort-level bias-corrected
mean. Use this for absolute-thickness claims.

---

## 6. Group-level significance testing

`group_significance.yaml` has three sections:

### 6.1. Groups

```yaml
groups:
  Control:
    n: 4
    mean_of_means: 350.4
    std_of_means: 28.1
  Collagen:
    n: 8
    mean_of_means: 412.5
    std_of_means: 35.6
```

This is the per-group sample-weighted summary. Each group's
`mean_of_means` is the mean of the (raw, not censored) per-sample means
within that group.

### 6.2. Omnibus: Kruskal-Wallis

```yaml
omnibus:
  test: Kruskal-Wallis
  statistic: 7.42
  p_value: 0.024
  groups_tested: [Control, Podocin, Collagen]
```

Non-parametric test of "do any of these groups have different
distributions?". Robust to non-normality (which thickness data is) and
robust to censoring (rank-based). If `p_value < 0.05`, at least one
group differs from the others.

### 6.3. Pairwise: Mann-Whitney U + Bonferroni + Cliff's delta

```yaml
pairwise:
- a: Control
  b: Podocin
  a_mean: 350.4
  b_mean: 378.2
  U: 9.0
  p_raw: 0.040
  p_bonferroni: 0.119    # × 3 pairs
  cliffs_delta: 0.62
  magnitude: large
  significant_05: false
```

For each pair of groups: a Mann-Whitney U test (non-parametric
two-sided test of equal distributions) and Cliff's delta as the effect
size. Bonferroni correction multiplies the raw p by the number of pairs
tested. `significant_05` is `p_bonferroni < 0.05`. Cliff's delta
magnitudes follow Romano et al. 2006: |d| < 0.147 negligible, < 0.330
small, < 0.474 medium, ≥ 0.474 large.

---

## 7. Visualisations

### 7.1. Comparative box plot — `comparative_box_plot.png`

Standard box plot across samples. Box = IQR (P25–P75), whiskers = Tukey
1.5×IQR fences, diamond = mean.

**Previously**: the "box" plotted P5 / P95 — the middle 90% of data —
labeled as if it were IQR. This was misleading by convention; a reader
expects the box to show the middle 50%. Now corrected to actual IQR.

### 7.2. Per-sample histograms

For each sample, histograms at 10 / 20 / 50 / 100 bins of the cleaned
voxel-level thickness data.

**Known limitation**: bin edges differ across samples (each uses its
own min/max). Histograms can't be directly overlaid. For cross-sample
distribution comparison, see the comparative box plot or violin plot.

### 7.3. Cylindrical (polar) analysis

For each sample, polar-coordinate analysis: thickness is averaged in
angular bins around the volume centre. Empty bins (no GBM data at that
angle) are now **NaN** (not 0 as previously — the old behaviour
silently dragged means down and made empty bins indistinguishable from
genuinely thin GBM). NaN bins are **dropped** before plotting and the
filled polygon is closed only between adjacent valid bins, so the
fill region is bounded entirely by real data (the old behaviour let
Plotly's `fill='toself'` close the polygon through empty bins, drawing
phantom values at angles where there was nothing). The radial axis is
also fixed at 0 → 1200 nm so polar plots are comparable across samples.

### 7.4. Top-down + combined view

Maximum intensity projection along Z, with optional polygon overlay of
the cylindrical-average thickness. The colour scale is **fixed at
0 → `HARD_THICKNESS_LIMIT_NM` (1200 nm)** across every sample, so a
given colour represents the same physical thickness everywhere — the
heatmaps are directly comparable side-by-side. Previously the colour
scale auto-ranged per sample, which made the same shade of yellow
mean different thicknesses on different samples.

---

### 7.5. Publication figure set — `publication/` subdirectory

In addition to the per-sample PNGs above (which are kept unchanged), the
pipeline writes a curated, consistently-styled set of figures into
`<tag>_stats/publication/`, each in **PNG + PDF + SVG** (raster for
quick-look, vector for manuscript embedding). All use a colourblind-safe
palette (Okabe-Ito), large fonts, and a fixed 0 → 1200 nm thickness axis
so panels are directly comparable.

| File (×3 formats) | What it shows |
|---|---|
| `thickness_violin_per_sample` | One violin per sample, coloured by disease group, embedded box (median + IQR). The previously-promised-but-never-written violin figure. |
| `thickness_violin_per_group` | Violins pooled by disease group (Control / Podocin / Collagen) with **significance brackets** (★/★★/★★★) drawn for pairs passing Bonferroni < 0.05. |
| `comparative_box_publication` | Restyled per-sample box plot; the bias-corrected (censored) mean is overlaid as a hollow diamond next to each box. |
| `qq_lognormal_grid` | Per-sample lognormal QQ plots (log-thickness vs normal quantiles) in a grid; points on the diagonal ⇒ lognormal — the visual check behind the censored-MLE assumption. |
| `normality_report.yaml` | Shapiro-Wilk on raw vs log-transformed thickness + skewness/kurtosis per sample (see §7.6). |

Voxel data is deterministically subsampled (seeded) for the violins
(≤20k points/sample) and QQ plots (≤2k points/sample) to keep file sizes
manageable; the distribution shape is preserved.

### 7.6. Normality / lognormal-assumption check — `normality_report.yaml`

The censored-MLE thickness correction (§4) assumes the per-voxel
thickness is **lognormal**. This report validates that assumption per
sample. It runs on the **less-processed data** — post hard-cap and COL4
mask, but **before** the IQR outlier removal — because the latent
distribution the censored MLE models is the *untruncated* thickness;
testing the IQR-clipped `data_clean` would make even a perfectly
lognormal sample fail (you've cut off the tails that make it look
lognormal). For each sample it runs the Shapiro-Wilk test on both the raw
and the log-transformed thickness (subsampled to N≤5000, Shapiro's valid
range), plus skewness and excess kurtosis:

```yaml
samples:
- sample_name: NCWM.AUY380.Series004
  n: 50000
  shapiro_W_raw: 0.926      # raw data: far from normal
  shapiro_W_log: 0.9995     # log data: essentially normal
  skewness_raw: 1.25        # right-skewed
  skewness_log: 0.03        # symmetric after log
  lognormal_preferred: true # W_log > W_raw
summary:
  n_samples_tested: 15
  n_lognormal_preferred: 15
  fraction_lognormal_preferred: 1.0   # ← close to 1.0 validates the lognormal MLE
```

If `fraction_lognormal_preferred` is close to 1.0, the lognormal model is
well-supported and the censored-MLE correction is sound. If it's low, the
lognormal assumption is questionable for those samples — inspect the
`qq_lognormal_grid` figure and treat their `censored_mean` cautiously.

**Reading the QQ grid as the arbiter**: if points fall on the diagonal in
the *middle* and only deviate at the extreme ends, that's residual
truncation/censoring — the lognormal model is still fine. A systematic
S-curve or a kink *through the body* of the plot indicates genuine
non-lognormality (e.g. a bimodal thin/thick GBM population), in which
case the parametric `censored_mean` should be reported with caution and
the non-parametric rank statistics (§6) preferred for that sample.

---

## 8. Per-sample annotations in `metadata.txt`

The metadata file lists samples with their group, lists samples flagged
for high PSF clamp (kept in aggregate, but flagged for the reader),
documents the parameters used (hard thickness cap, uniform mask
percentile, IQR convention), and the generated files. Read it first —
it gives the context for everything else.

---

## 9. What's still bias / open problems

- **Lognormal assumption**: the censored-MLE correction assumes
  thickness is lognormal. For most biological thickness measurements
  this holds, but if a sample has a bimodal distribution (e.g. two
  populations of GBM thickness in disease states), the lognormal fit
  is misleading. A QQ plot per sample would diagnose this; not
  currently in the pipeline.
- **Angle-dependent PSF threshold**: the censored MLE uses the lateral
  PSF (149 nm) as the threshold. The actual threshold per voxel
  depends on surface-normal angle:
  `√(PSF_lat² cos²α + PSF_ax² sin²α)`. Since this varies from 149 nm
  (face-on) to 434 nm (edge-on), the effective threshold is
  **angle-weighted** in a way the current correction doesn't model. A
  more rigorous fit would use the per-voxel angle distribution as a
  known nuisance variable.
- **Voxel-level correlation**: the MLE treats voxels as independent
  samples. Adjacent voxels in a 3D thickness map are spatially
  correlated, which means the effective sample size for the MLE is
  smaller than `n_observed`. The point estimate is unaffected but the
  reported confidence interval is too narrow. For publication, prefer
  the per-sample mean (n=samples) over the per-voxel mean (n=voxels).
- **Group sizes**: with current v6 inference (all-Collagen test set),
  cross-group testing is degenerate (only one group). When the labeled
  data spans all three groups (e.g. running on the training set via
  `gbm.py labels-as-pred`), the significance tests become informative.

---

## 10. Quick reference

| Question | Field to read |
|---|---|
| What's the mean GBM thickness for sample X? | `summary_statistics.npz` → `censored_mean` (bias-corrected). |
| What's the mean GBM thickness for the cohort? | `sample_weighted_aggregate.yaml` → `censored.mean_of_sample_means`. |
| Is Collagen GBM thicker than Control? | `group_significance.yaml` → `pairwise[Control,Collagen].significant_05` and `cliffs_delta`. |
| Why is this sample's mean suspicious? | `metadata.txt` → "Flagged (high PSF clamp)" section. |
| How was outlier removal done? | Per-sample: Tukey IQR (P25/P75 with 1.5× fences). Both lower and upper. |
| What was masked out before stats? | Voxels above the 1200 nm hard cap; voxels above P95 of the COL4 channel. |
| Are heatmaps comparable across samples? | Yes — colour scale fixed at 0 → 1200 nm. |
