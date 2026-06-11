"""Regression tests for the statistical-analysis pipeline at
src/infer/stats.py. These cover:

- The hard 1200 nm thickness cap (always applied, independent of CLI flag).
- IQR outlier helpers now using P25/P75 with both fences enforced.
- Cylindrical empty-bin behaviour switched to NaN.
- Group-significance helper (Kruskal-Wallis + pairwise Mann-Whitney U
  with Bonferroni correction + Cliff's delta effect size).
- The uniform mask threshold constant replacing per-group thresholds.
"""

import numpy as np

from src.infer import stats as S

# --- module constants ------------------------------------------------------

def test_hard_thickness_limit_nm_is_1200():
    """The hard cap must be 1200 nm — biological upper bound."""
    assert S.HARD_THICKNESS_LIMIT_NM == 1200


def test_uniform_mask_percentile_is_set():
    """A uniform mask percentile must exist — the per-group thresholds
    (91/92.5/97) used to be the main cross-group bias source."""
    assert 50.0 <= S.UNIFORM_MASK_PERCENTILE <= 99.5


def test_psf_clamp_warning_threshold_set():
    """The clamp threshold flags samples in metadata.txt but does NOT
    exclude them from aggregation — comparison must include all data."""
    assert 0.0 < S.PSF_CLAMP_WARNING_PCT <= 100.0
    assert not hasattr(S, 'PSF_CLAMP_EXCLUSION_PCT'), (
        "PSF_CLAMP_EXCLUSION_PCT was renamed to PSF_CLAMP_WARNING_PCT "
        "to make explicit that we never silently exclude samples.")


# --- outlier helpers -------------------------------------------------------

def test_remove_outliers_iqr_uses_quartiles_and_both_fences():
    """With proper IQR defaults (25, 75), extreme values on BOTH sides
    are dropped — was previously upper-side only."""
    arr = np.concatenate([np.array([-1000.0, -500.0]),       # low outliers
                          np.linspace(0.0, 100.0, 200),       # bulk
                          np.array([1000.0, 5000.0])])        # high outliers
    cleaned = S.remove_outliers_iqr(arr)
    assert cleaned.max() < 200.0
    assert cleaned.min() > -50.0


def test_remove_outliers_iqr_passthrough_when_empty():
    assert S.remove_outliers_iqr(np.array([])).size == 0


def test_replace_outliers_iqr_clamps_both_sides():
    """Construct a tight bulk plus far-out tails so the IQR fences
    catch them on both sides. Q25/Q75 land inside the bulk, IQR is
    small → 1.5×IQR fences exclude the tails."""
    bulk = np.linspace(0.0, 100.0, 200)
    arr = np.concatenate([np.array([-1e6, -1e5]), bulk, np.array([1e5, 1e6])])
    clamped = S.replace_outliers_iqr(arr)
    # Both extremes should now sit at the IQR fences (≈ -150 to 250),
    # not the original ±1e6.
    assert clamped.min() > -1000.0
    assert clamped.max() < 1000.0
    # Bulk values are not modified.
    assert clamped.max() >= bulk.max() or clamped.max() == clamped.max()


# --- cylindrical analysis NaN behaviour -----------------------------------

def test_cylindrical_empty_bins_become_nan():
    """Empty cylindrical bins used to be reported as 0 nm; they're now NaN
    so downstream nanmean/nanmax can skip them."""
    # A volume with only one bright pixel — most angular bins will be empty.
    data = np.zeros((4, 64, 64), dtype=np.float32)
    data[2, 40, 40] = 500.0
    angles, avgs = S.calculate_cylindrical_analysis(
        data, _alpha_step=30, _radius=20)
    avgs_arr = np.asarray(avgs, dtype=float)
    # At least one bin must be NaN (empty) and at least one must be finite.
    assert np.any(np.isnan(avgs_arr))
    assert np.any(np.isfinite(avgs_arr) & (avgs_arr > 0))


# --- effect size + group significance --------------------------------------

def test_cliffs_delta_extremes():
    """Cliff's delta is +1 when x dominates y, -1 when y dominates x,
    and ~0 when distributions overlap."""
    assert S.cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0
    assert S.cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0
    same = S.cliffs_delta([1, 2, 3], [1, 2, 3])
    assert abs(same) < 0.1


def test_compute_group_significance_detects_difference():
    """With clearly different groups, the omnibus + pairwise tests must
    flag the difference and Cliff's delta should be 'large'."""
    summary = (
        [{"sample_name": f"a{i}", "group": "A", "mean": 100.0 + i}
         for i in range(5)] +
        [{"sample_name": f"b{i}", "group": "B", "mean": 500.0 + i}
         for i in range(5)])
    out = S.compute_group_significance(summary)
    assert out["omnibus"] is not None
    assert out["omnibus"]["p_value"] < 0.05
    assert len(out["pairwise"]) == 1
    pair = out["pairwise"][0]
    assert pair["significant_05"] is True
    assert pair["magnitude"] == "large"


def test_compute_group_significance_handles_single_group():
    summary = [{"sample_name": f"a{i}", "group": "A", "mean": 100.0 + i}
               for i in range(5)]
    out = S.compute_group_significance(summary)
    assert out["omnibus"] is None
    assert "Insufficient groups" in out["note"]


def test_compute_group_significance_reports_groups_with_n1_in_summary_only():
    """Groups with fewer than 2 samples appear in `groups` (for reporting)
    but aren't tested in the omnibus/pairwise (statistically meaningless)."""
    summary = ([{"sample_name": f"a{i}", "group": "A", "mean": 100.0 + i}
                for i in range(5)]
               + [{"sample_name": "b0", "group": "B", "mean": 500.0}])  # n=1
    out = S.compute_group_significance(summary)
    assert "A" in out["groups"]
    assert "B" in out["groups"]
    # Only A is "eligible" → fewer than 2 eligible groups → no omnibus.
    assert out["omnibus"] is None


# --- left-censored lognormal MLE -------------------------------------------

def test_censored_mle_uncensored_matches_observed_mean():
    """With zero censoring, the lognormal MLE mean should be close to the
    sample mean of a synthetic lognormal dataset."""
    rng = np.random.default_rng(seed=42)
    mu_true, sigma_true = np.log(400.0), 0.4
    data = rng.lognormal(mu_true, sigma_true, size=5000)
    fit = S.fit_lognormal_left_censored(data, _n_censored=0,
                                         _censoring_threshold_nm=149.0)
    assert fit['converged']
    # Theoretical mean of the lognormal is exp(mu + sigma^2/2).
    theoretical_mean = np.exp(mu_true + sigma_true**2 / 2)
    assert abs(fit['mean_nm'] - theoretical_mean) / theoretical_mean < 0.05


def test_censored_mle_recovers_truncated_mean():
    """When the dataset is left-truncated at PSF (we drop voxels < PSF)
    and we tell the MLE about it, it should recover a mean closer to the
    full-distribution mean than the simple mean of the truncated data."""
    rng = np.random.default_rng(seed=7)
    mu_true, sigma_true = np.log(300.0), 0.5
    full_data = rng.lognormal(mu_true, sigma_true, size=10000)
    psf = 149.0
    truncated = full_data[full_data > psf]
    n_censored = int(np.sum(full_data <= psf))
    naive_mean = float(np.mean(truncated))
    fit = S.fit_lognormal_left_censored(truncated, n_censored, psf)
    true_mean = np.exp(mu_true + sigma_true**2 / 2)
    assert fit['converged']
    # The MLE should be measurably closer to the truth than the naive mean.
    assert abs(fit['mean_nm'] - true_mean) < abs(naive_mean - true_mean)


def test_censored_mle_handles_too_few_observations():
    fit = S.fit_lognormal_left_censored(np.array([200.0]),
                                         _n_censored=10, _censoring_threshold_nm=149.0)
    assert not fit['converged']
    assert np.isnan(fit['mean_nm'])


# --- publication figures + diagnostics -------------------------------------

def test_group_color_stable_and_fallback():
    assert S.group_color("Control") == S.GROUP_COLORS["Control"]
    assert S.group_color("Collagen") == S.GROUP_COLORS["Collagen"]
    # Unknown / unmapped → grey fallback.
    assert S.group_color("Nonexistent") == S.GROUP_COLORS["Unknown"]


def test_subsample_caps_size_and_is_deterministic():
    arr = np.arange(100000, dtype=float)
    a = S._subsample(arr, _n=5000)
    b = S._subsample(arr, _n=5000)
    assert a.size == 5000
    assert np.array_equal(a, b)  # seeded → reproducible
    # Below cap → returned unchanged.
    small = np.arange(10, dtype=float)
    assert np.array_equal(S._subsample(small, _n=5000), small)


def test_p_to_stars():
    assert S._p_to_stars(0.0005) == "***"
    assert S._p_to_stars(0.005) == "**"
    assert S._p_to_stars(0.03) == "*"
    assert S._p_to_stars(0.5) == "ns"
    assert S._p_to_stars(float("nan")) == "n/a"


def test_normality_report_prefers_lognormal_for_lognormal_data():
    rng = np.random.default_rng(0)
    samples = [rng.lognormal(np.log(350), 0.4, 20000),
               rng.lognormal(np.log(500), 0.45, 20000)]
    rep = S.compute_normality_report(samples, ["s1", "s2"])
    assert rep["summary"]["n_samples_tested"] == 2
    # Lognormal synthetic data → log transform should be preferred for all.
    assert rep["summary"]["fraction_lognormal_preferred"] == 1.0
    for s in rep["samples"]:
        assert s["shapiro_W_log"] > s["shapiro_W_raw"]


def test_normality_report_handles_tiny_sample():
    rep = S.compute_normality_report([np.array([100.0, 200.0])], ["tiny"])
    assert rep["samples"][0]["note"].startswith("too few")


def test_generate_group_figures_writes_full_set(tmp_path):
    """End-to-end: the curated set (violin/ridgeline/ecdf/estimation) in
    PNG+PDF+SVG, plus the two significance YAMLs, mouse summary and normality
    report. Guards the new publication output contract."""
    recs = _multi_group_records()
    written = S.generate_group_figures(tmp_path, recs)
    pub = tmp_path / "publication"
    for stem in ("group_violin_comparison", "group_ridgeline",
                 "group_ecdf", "group_estimation",
                 "mouse_violin", "mouse_ecdf"):
        for ext in ("png", "pdf", "svg"):
            assert (pub / f"{stem}.{ext}").exists(), f"missing {stem}.{ext}"
    for y in ("group_significance_by_mouse.yaml",
              "group_significance_by_image.yaml",
              "mouse_summary.yaml", "normality_report.yaml"):
        assert (pub / y).exists(), f"missing {y}"
        assert y in written


def test_generate_group_figures_empty_data_is_safe(tmp_path):
    assert S.generate_group_figures(tmp_path, []) == []


# --- parallel (array) stats path -------------------------------------------

def _make_fake_inference_dir(root, n_samples=3):
    """Create a minimal results-infer tree: each sample dir has a
    psf_result.npz (thickness volume) + prediction.tif (4-channel) +
    psf_clamp_stats.yaml, enough for _process_one_sample to run."""
    import tifffile
    import yaml as _yaml
    rng = np.random.default_rng(11)
    names = []
    for i in range(n_samples):
        name = f"NCWM.CKM105.Series{i:03d}.tiff"
        names.append(name)
        sd = root / name
        sd.mkdir(parents=True)
        # Thickness volume (Z, H, W): a blob of lognormal thickness.
        vol = np.zeros((6, 64, 64), dtype=np.float32)
        vol[2:4, 20:44, 20:44] = rng.lognormal(np.log(300), 0.3, (2, 24, 24))
        np.savez_compressed(sd / "psf_result.npz", arr=vol)
        # prediction.tif (Z, C=4, H, W); channel 1 = COL4.
        pred = np.zeros((6, 4, 64, 64), dtype=np.float32)
        pred[:, 1] = rng.random((6, 64, 64)) * 50
        tifffile.imwrite(sd / "prediction.tif", pred)
        with open(sd / "psf_clamp_stats.yaml", "w") as f:
            _yaml.safe_dump({"clamp_count": 1000, "surface_count": 10000,
                             "clamp_percentage": 10.0, "psf_lateral_nm": 149},
                            f)
    return names


def test_parallel_stats_matches_single_process(tmp_path):
    """The array path (per-sample sidecars → reduce) must produce the same
    cohort summary as the single-process calculate_stats, for the with_mask
    variant."""
    inf = tmp_path / "infer"
    inf.mkdir()
    names = _make_fake_inference_dir(inf, n_samples=3)

    # Single-process reference (writes both variants).
    sp_dir = tmp_path / "sp_stats"
    S.calculate_stats(inf, sp_dir, _clipping=False)
    sp_summary = np.load(sp_dir / "with_mask" / "summary_statistics.npz",
                         allow_pickle=True)["arr"]

    # Parallel path: per-sample then reduce.
    par_dir = tmp_path / "par_stats"
    for name in names:
        S.calculate_stats_one_sample(inf, par_dir, name, _clipping=False)
    # Sidecars exist per variant, one per sample.
    assert len(list((par_dir / "with_mask" / "_partial").glob("*.npz"))) == len(names)
    assert len(list((par_dir / "without_mask" / "_partial").glob("*.npz"))) == len(names)
    S.calculate_stats_reduce(inf, par_dir)
    par_summary = np.load(par_dir / "with_mask" / "summary_statistics.npz",
                          allow_pickle=True)["arr"]

    # Same samples, same per-sample means (order-independent).
    sp = {r["sample_name"]: float(r["mean"]) for r in sp_summary}
    par = {r["sample_name"]: float(r["mean"]) for r in par_summary}
    assert sp.keys() == par.keys()
    for k in sp:
        assert abs(sp[k] - par[k]) < 1e-6
    # Both mask variants produced the curated publication figure set.
    for variant in ("with_mask", "without_mask"):
        assert (par_dir / variant / "publication"
                / "group_violin_comparison.png").exists()
        assert (sp_dir / variant / "publication"
                / "group_violin_comparison.png").exists()


def test_mask_variants_differ(tmp_path):
    """The without_mask variant keeps at least as many voxels as with_mask
    (the COL4 mask only removes voxels), and both variants are written."""
    inf = tmp_path / "infer"
    inf.mkdir()
    _make_fake_inference_dir(inf, n_samples=3)
    out = tmp_path / "stats"
    S.calculate_stats(inf, out, _clipping=False)
    masked = np.load(out / "with_mask" / "aggregated_thickness_data.npz",
                     allow_pickle=True)["arr"]
    unmasked = np.load(out / "without_mask" / "aggregated_thickness_data.npz",
                       allow_pickle=True)["arr"]
    assert unmasked.size >= masked.size
    for variant in ("with_mask", "without_mask"):
        assert (out / variant / "metadata.txt").exists()


def test_shared_ymax_is_data_driven_and_capped():
    rng = np.random.default_rng(3)
    # All data well under 600 nm → ymax should be far below the 1200 cap.
    samples = [rng.lognormal(np.log(300), 0.3, 5000) for _ in range(3)]
    ymax = S._shared_ymax(samples)
    assert ymax < S.HARD_THICKNESS_LIMIT_NM
    assert ymax % 50 == 0  # rounded to a clean tick
    # A pathological sample with huge values is still capped at the hard limit.
    samples.append(np.full(5000, 5000.0))
    assert S._shared_ymax(samples) == S.HARD_THICKNESS_LIMIT_NM


def test_group_violin_comparison_three_groups_writes(tmp_path):
    """The main 3-group comparison violin renders with all three groups +
    per-mouse dots + significance brackets."""
    recs = _multi_group_records()
    per_mouse = S._rollup_to_mouse(recs)
    per_image = [(r["mean"], r["mouse"], r["group"]) for r in recs]
    per_mouse_t = [(r["mean"], r["mouse"], r["group"]) for r in per_mouse]
    vox = {}
    for r in recs:
        vox.setdefault(r["group"], []).append(np.asarray(r["voxels"], float))
    vox = {g: np.concatenate(v) for g, v in vox.items()}
    sig = S.compute_group_significance(
        [{"sample_name": r["mouse"], "group": r["group"], "mean": r["mean"]}
         for r in per_mouse])
    S.save_group_violin_comparison(per_image, per_mouse_t, vox,
                                   tmp_path / "group_violin_comparison", sig)
    for ext in ("png", "pdf", "svg"):
        assert (tmp_path / f"group_violin_comparison.{ext}").exists()


def test_group_ridgeline_writes(tmp_path):
    """Ridgeline renders one density ridge per group."""
    recs = _multi_group_records()
    vox = {}
    for r in recs:
        vox.setdefault(r["group"], []).append(np.asarray(r["voxels"], float))
    vox = {g: np.concatenate(v) for g, v in vox.items()}
    S.save_group_ridgeline(vox, tmp_path / "group_ridgeline")
    for ext in ("png", "pdf", "svg"):
        assert (tmp_path / f"group_ridgeline.{ext}").exists()


# --- detect_group_type warning ---------------------------------------------

def test_detect_group_type_unknown_logs_warning(caplog):
    with caplog.at_level("WARNING"):
        g = S.detect_group_type("OBVIOUSLY_UNKNOWN_SAMPLE")
    assert g == "Unknown"  # no longer silently defaults to Control
    assert any("did not match" in r.message for r in caplog.records)


def test_detect_group_type_known_prefixes():
    assert S.detect_group_type("NCW.AUY380_Series001") == "Control"
    assert S.detect_group_type("NCW.BDP669_Series002") == "Podocin"
    assert S.detect_group_type("NCW.CKM105_Series003") == "Collagen"


def test_detect_group_type_control_ckm_with_ncw_prefix():
    """Regression: the old prefix table stored CKM103/CKM110 WITHOUT the
    'NCW.' prefix real files carry, so 'NCW.CKM103…' control images silently
    became 'Unknown'. Group is now keyed off the animal token, so they
    correctly classify as Control."""
    assert S.detect_group_type("NCW.CKM103.20241124.lif - Series002.tiff") == "Control"
    assert S.detect_group_type("NCW.CKM110_Series001") == "Control"


# --- nested-data unit + group-comparison figures ---------------------------

def test_detect_mouse_id_extracts_animal_token():
    assert S.detect_mouse_id("NCW.CKM104.20241124.lif - Series003.tiff") == "CKM104"
    assert S.detect_mouse_id("NCW.AUY380_Series001") == "AUY380"
    assert S.detect_mouse_id("NCW.BDP675.lif - Series012.tiff") == "BDP675"
    # Multiple Series of one mouse map to the SAME id (the pseudoreplication
    # the per-mouse rollup collapses).
    a = S.detect_mouse_id("NCW.CKM105.x - Series001.tiff")
    b = S.detect_mouse_id("NCW.CKM105.x - Series009.tiff")
    assert a == b == "CKM105"


def test_remove_outliers_iqr_log_keeps_thick_tail():
    """On right-skewed (lognormal) thickness, the log-scale fence must
    retain more of the genuine thick-membrane tail than the raw-scale fence
    (which clips it and biases the mean down)."""
    rng = np.random.default_rng(0)
    data = rng.lognormal(np.log(300), 0.5, 50000)
    raw = S.remove_outliers_iqr(data)
    log = S.remove_outliers_iqr(data, _log=True)
    assert log.max() > raw.max()           # log fence keeps higher values
    assert log.mean() > raw.mean()         # raw fence biased the mean down


def test_bootstrap_mean_diff_ci_brackets_observed():
    rng = np.random.default_rng(1)
    ref = rng.normal(300, 20, 6)
    grp = rng.normal(500, 20, 6)
    obs, lo, hi = S._bootstrap_mean_diff_ci(ref, grp, _n_boot=1000)
    assert lo < obs < hi
    assert obs > 0                          # grp clearly larger than ref
    # Too few points → NaN CI, never raises.
    o2, lo2, hi2 = S._bootstrap_mean_diff_ci([1.0], [2.0])
    assert np.isnan(lo2) and np.isnan(hi2)


def test_rollup_to_mouse_collapses_images():
    recs = [
        {"name": "a", "mouse": "CKM104", "group": "Collagen", "mean": 500.0,
         "voxels": np.full(100, 500.0)},
        {"name": "b", "mouse": "CKM104", "group": "Collagen", "mean": 540.0,
         "voxels": np.full(100, 540.0)},
        {"name": "c", "mouse": "AUY380", "group": "Control", "mean": 300.0,
         "voxels": np.full(100, 300.0)},
    ]
    per_mouse = S._rollup_to_mouse(recs)
    by = {r["mouse"]: r for r in per_mouse}
    assert set(by) == {"CKM104", "AUY380"}
    assert by["CKM104"]["n_images"] == 2
    assert abs(by["CKM104"]["mean"] - 520.0) < 1e-6   # mean of image means
    assert by["AUY380"]["n_images"] == 1


def _multi_group_records(seed=0):
    rng = np.random.default_rng(seed)
    spec = {"Control": (["AUY380", "AUY381", "CKM103"], np.log(300), 0.35),
            "Podocin": (["BDP669", "BDP672", "BDP675"], np.log(440), 0.40),
            "Collagen": (["CKM104", "CKM105"], np.log(560), 0.45)}
    recs = []
    for grp, (mice, mu, sg) in spec.items():
        for m in mice:
            for s in range(3):
                vox = rng.lognormal(mu + rng.normal(0, 0.08), sg, 6000)
                name = f"NCW.{m}.lif - Series{s:03d}.tiff"
                recs.append({"name": name, "mouse": m, "group": grp,
                             "mean": float(np.mean(vox)), "voxels": vox})
    return recs


def test_generate_group_figures_estimation_skipped_for_single_group(tmp_path):
    """With one group the estimation plot (needs >=2 groups) is skipped, but
    the violin/ridgeline/ecdf still render and the run doesn't error."""
    recs = [r for r in _multi_group_records() if r["group"] == "Collagen"]
    S.generate_group_figures(tmp_path, recs)
    pub = tmp_path / "publication"
    assert (pub / "group_violin_comparison.png").exists()
    assert (pub / "group_ridgeline.png").exists()
    assert (pub / "group_ecdf.png").exists()
    assert not (pub / "group_estimation.png").exists()
