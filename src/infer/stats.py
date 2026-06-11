# Statistics + visualization for inference results.
# Moved out of src/utils/misc.py during the Phase 3 split.

import json
import logging
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tifffile
import yaml
from plotly.subplots import make_subplots
from scipy.ndimage import binary_opening
from scipy.optimize import minimize
from scipy.stats import (
    gaussian_kde,
    kruskal,
    kurtosis,
    mannwhitneyu,
    norm,
    shapiro,
    skew,
)

# Hard biological upper limit on GBM thickness, in nanometres.
# Values above this are treated as measurement artefacts (PSF residuals,
# segmentation noise) and masked out *unconditionally* at the very start
# of `calculate_stats`, independent of the optional --clipping CLI flag.
# 1200 nm is the documented upper end of plausible GBM thickness in
# pathology literature; anything beyond is not biology.
HARD_THICKNESS_LIMIT_NM = 1200

# PSF-clamp activation threshold (% of surface voxels) above which a
# sample is *flagged* in metadata.txt as having a mean-thickness bias.
# Samples are NEVER silently excluded from aggregation or significance
# testing — the comparison includes all data and the user reads the
# flag alongside the result. Empirically the Collagen pathology cohort
# runs 20-44% clamp on every sample (thin membrane → many sub-PSF
# measurements), so this is annotation, not a filter.
PSF_CLAMP_WARNING_PCT = 20.0

# Mask threshold used across all sample groups when masking high-COL4
# (capillary lumen / glomerular boundary) voxels before computing GBM
# thickness statistics. Was previously 91/92.5/97 per group, which
# produced systematic measurement bias mimicking biological differences.
# Use a single value uniformly so cross-group comparisons are valid.
UNIFORM_MASK_PERCENTILE = 95.0

# --- Publication-figure styling -------------------------------------------
# Stable disease-group → colour mapping for the publication figures, so the
# same group is always the same colour across panels. A muted, modern
# (seaborn-"deep"-style) palette — softer than the bright Okabe-Ito set and
# still distinguishable under the common colour-vision deficiencies
# (blue / amber / green). Unknown falls back to grey.
GROUP_COLORS = {
    "Control":  "#4C72B0",  # muted blue
    "Podocin":  "#DD8452",  # muted amber
    "Collagen": "#55A868",  # muted green
    "Unknown":  "#9C9C9C",  # grey
}

# Ink colour for axes / outlines / text — a soft near-black, not pure #000,
# for the cleaner editorial look of the new figures.
PUB_INK = "#2B2B2B"

# Publication font sizes (points). Tuned for single-column journal figures
# rendered at ~1200 px wide.
PUB_FONT_FAMILY = "Arial, Helvetica, sans-serif"
PUB_TITLE_SIZE = 22
PUB_AXIS_TITLE_SIZE = 20
PUB_TICK_SIZE = 16
PUB_ANNOT_SIZE = 14

# Vector + raster formats emitted for every publication figure. PDF/SVG are
# scalable (embeddable in LaTeX / Illustrator); PNG is a quick-look raster.
PUB_FIGURE_FORMATS = ("png", "pdf", "svg")

# Raster up-scaling factor for the PNG exports. scale=4 quadruples the pixel
# density (≈ 4× the dots-per-figure in each axis) for crisp screen / print
# rasters. PDF and SVG are vector formats — resolution-independent — so the
# scale factor is only applied to PNG (and other raster formats).
PUB_PNG_SCALE = 4


def replace_outliers_iqr(arr, k=1.5, lower_p=25, upper_p=75,
                         lower_p_zero_iqr=2, upper_p_zero_iqr=98):
    """Tukey-style outlier clamping. Both lower and upper IQR fences are
    enforced (was previously upper-only). Defaults use the proper IQR
    quartiles (25, 75) — the old defaults of (5, 95) gave a "fence" at
    ~6.6σ on normal data, effectively a no-op.
    """
    original_size = arr.size
    q1 = np.percentile(arr, lower_p)
    q3 = np.percentile(arr, upper_p)
    iqr = q3 - q1
    if iqr == 0:
        logging.info("IQR is 0, using percentiles for outlier detection.")
        lower_bound = np.percentile(arr, lower_p_zero_iqr)
        upper_bound = np.percentile(arr, upper_p_zero_iqr)
    else:
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
    logging.info(f"Outlier replacement: q1={q1}, q3={q3}, iqr={iqr}, "
                 f"lower_bound={lower_bound}, upper_bound={upper_bound}")
    arr_copy = arr.copy()
    upper_mask = arr_copy > upper_bound
    lower_mask = arr_copy < lower_bound
    replaced_upper = int(np.sum(upper_mask))
    replaced_lower = int(np.sum(lower_mask))
    arr_copy[upper_mask] = upper_bound
    arr_copy[lower_mask] = lower_bound
    if original_size > 0:
        pct = (replaced_upper + replaced_lower) / original_size * 100
        logging.info(
            f"Clamped {replaced_lower} below and {replaced_upper} above "
            f"the IQR fences out of {original_size} values ({pct:.2f}%).")
    return arr_copy


def remove_outliers_iqr(arr, k=1.5, lower_p=25, upper_p=75, _log=False):
    """Drop voxels outside the Tukey IQR fences. Both lower and upper
    fences are now enforced (was previously upper-only). Defaults use the
    proper IQR quartiles (25, 75) — the old defaults of (5, 95) gave a
    fence at ~6.6σ on normal data, effectively a no-op.

    With ``_log=True`` the fences are computed on the LOG of the (strictly
    positive) values. GBM thickness is right-skewed / lognormal, so a
    symmetric Tukey fence on the raw scale preferentially clips the upper
    tail — i.e. genuinely thick membrane — which biases the mean down and
    shrinks the variance. Working on the log scale makes the fence
    multiplicative (symmetric in the geometry of a lognormal), so only true
    outliers in either tail are dropped. Use this path for thickness."""
    original_size = len(arr)
    if original_size == 0:
        return arr
    if _log:
        work = arr[arr > 0]
        if work.size == 0:
            logging.warning("remove_outliers_iqr(_log=True): no positive "
                            "values to work with.")
            return work
        logw = np.log(work)
        q1, q3 = np.percentile(logw, [lower_p, upper_p])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        clean_arr = work[(logw >= lower_bound) & (logw <= upper_bound)]
        removed_count = original_size - len(clean_arr)
        logging.info(
            "Outlier removal (log scale): fences=exp[%.3f, %.3f] = "
            "[%.2f, %.2f] nm; removed %d/%d (%.2f%%).",
            lower_bound, upper_bound, np.exp(lower_bound), np.exp(upper_bound),
            removed_count, original_size,
            (removed_count / original_size) * 100)
        return clean_arr
    q1 = np.percentile(arr, lower_p)
    q3 = np.percentile(arr, upper_p)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    logging.info(f"Outlier removal: q1={q1}, q3={q3}, iqr={iqr}, "
                 f"lower_bound={lower_bound}, upper_bound={upper_bound}")
    clean_arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]
    removed_count = original_size - len(clean_arr)
    percentage_removed = (removed_count / original_size) * 100
    logging.info(f"Removed {removed_count} outliers out of {original_size} "
                 f"values ({percentage_removed:.2f}%).")
    return clean_arr


def save_histogram(_array, _title, _path, _bins):
    if _array.size < 2 or np.max(_array) == np.min(_array):
        bin_config = dict(nbinsx=_bins)
    else:
        bin_size = (np.max(_array) - np.min(_array)) / _bins
        bin_config = dict(xbins=dict(
            start=np.min(_array), end=np.max(_array), size=bin_size))

    fig = go.Figure(data=[go.Histogram(
        x=_array, **bin_config,
        marker=dict(line=dict(color='black', width=1))
    )])
    fig.update_layout(title=_title, xaxis_title='Value', yaxis_title='Frequency')
    logging.info(f"Saving histogram: {_path}")
    fig.write_image(_path, width=1400, height=1000)  # requires kaleido


def save_polar_plot(_angles, _thickness_values, _title, _path,
                    _max_thickness=HARD_THICKNESS_LIMIT_NM):
    """Polar plot of average thickness per angular bin.

    NaN values mark angles with no GBM data and are DROPPED before
    plotting — previously Plotly's ``fill='toself'`` closed the polygon
    through those bins, drawing fake values at angles where there was
    nothing. Now the polygon is built only from valid bins and closed
    cleanly between adjacent valid neighbours; the radial axis is
    fixed at 0 → HARD_THICKNESS_LIMIT_NM so polar plots are directly
    comparable across samples.
    """
    angles_arr = np.asarray(_angles, dtype=float)
    values_arr = np.asarray(_thickness_values, dtype=float)
    valid = ~np.isnan(values_arr)
    angles_valid = angles_arr[valid]
    values_valid = values_arr[valid]

    fig = go.Figure()
    if values_valid.size >= 2:
        # Close the polygon explicitly using the FIRST VALID point, so
        # the fill region is bounded only by real data — never by the
        # centre or by NaN bridges.
        angles_closed = np.concatenate([angles_valid, [angles_valid[0]]])
        values_closed = np.concatenate([values_valid, [values_valid[0]]])
        fig.add_trace(go.Scatterpolar(
            r=values_closed, theta=angles_closed, mode='lines',
            line_color='blue', line_width=2, fill='toself',
            fillcolor='rgba(0, 0, 255, 0.1)'))
    elif values_valid.size == 1:
        # Degenerate: a single valid bin. Draw a marker, not a polygon.
        fig.add_trace(go.Scatterpolar(
            r=values_valid, theta=angles_valid, mode='markers',
            marker=dict(color='blue', size=8)))

    fig.update_layout(
        title=_title,
        polar=dict(
            radialaxis=dict(visible=True, title="Average Thickness (nm)",
                            tickangle=0, dtick=max(100, round(_max_thickness / 3 / 100) * 100),
                            range=[0, _max_thickness]),
            angularaxis=dict(visible=True, direction="clockwise",
                             period=360, dtick=45, rotation=0)))
    logging.info(f"Saving polar plot: {_path}")
    fig.write_image(_path, width=1400, height=1000)


def calculate_cylindrical_analysis(_data, _alpha_step, _radius):
    z_dim, y_dim, x_dim = _data.shape
    center_y, center_x = y_dim // 2, x_dim // 2

    angle_bins = np.arange(0, 360 + _alpha_step, _alpha_step)
    angles_for_plot = (angle_bins[:-1] + angle_bins[1:]) / 2
    binned_thickness_values = [[] for _ in range(len(angle_bins) - 1)]

    x_coords, y_coords = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    dx = x_coords - center_x
    dy = y_coords - center_y
    radius_map = np.sqrt(dx**2 + dy**2)
    angle_map_rad = np.arctan2(dy, dx)
    angle_map_deg = np.rad2deg(angle_map_rad)
    angle_map_deg[angle_map_deg < 0] += 360

    radius_mask = radius_map <= _radius

    angle_bin_indices = np.digitize(angle_map_deg, bins=angle_bins) - 1
    angle_bin_indices[angle_bin_indices == len(angle_bins) - 1] = len(angle_bins) - 2

    all_valid_thicknesses = []
    all_valid_angle_bins = []
    for z in range(z_dim):
        z_slice = _data[z, :, :]
        valid_mask = (z_slice > 0) & radius_mask
        if np.any(valid_mask):
            all_valid_thicknesses.append(z_slice[valid_mask])
            all_valid_angle_bins.append(angle_bin_indices[valid_mask])

    total_points_found = 0
    if all_valid_thicknesses:
        all_valid_thicknesses = np.concatenate(all_valid_thicknesses)
        all_valid_angle_bins = np.concatenate(all_valid_angle_bins)
        total_points_found = len(all_valid_thicknesses)

        total_points_binned_incremental = 0
        for i in range(len(binned_thickness_values)):
            bin_mask = all_valid_angle_bins == i
            if np.any(bin_mask):
                points_to_add = all_valid_thicknesses[bin_mask]
                binned_thickness_values[i].extend(points_to_add)
                total_points_binned_incremental += len(points_to_add)

        if total_points_found == total_points_binned_incremental:
            logging.debug(
                f"Point count validation successful: "
                f"Total points found ({total_points_found}) = "
                f"Total points incrementally binned ({total_points_binned_incremental}).")
        else:
            logging.warning(
                f"Point count mismatch: "
                f"Total points found ({total_points_found}) != "
                f"Total points incrementally binned ({total_points_binned_incremental}). "
                f"Some points were lost during binning.")

    avg_thickness_per_angle = []
    for i, values in enumerate(binned_thickness_values):
        if values:
            num_points = len(values)
            avg_thickness = float(np.mean(values))
            std_thickness = float(np.std(values, ddof=1)) if num_points > 1 else 0.0
            logging.debug(
                f"Cylindrical slice @ {angle_bins[i]}-{angle_bins[i+1]} deg: "
                f"Points={num_points}, Avg={avg_thickness:.2f}, Std={std_thickness:.2f}")
            avg_thickness_per_angle.append(avg_thickness)
        else:
            # NaN marks "no data" so downstream nanmean/nanmax skip it.
            # Previously we appended 0, which silently dragged means down
            # and made empty bins indistinguishable from genuinely-thin GBM.
            logging.debug(f"Cylindrical slice @ {angle_bins[i]}-{angle_bins[i+1]} deg: Points=0")
            avg_thickness_per_angle.append(np.nan)

    if avg_thickness_per_angle:
        angles_for_plot = np.append(angles_for_plot, angles_for_plot[0])
        avg_thickness_per_angle.append(avg_thickness_per_angle[0])

    return angles_for_plot, avg_thickness_per_angle


def save_top_down_view_aspect_ratio(_data, _title, _path,
                                    _lower_percentile_iqr, _upper_percentile_iqr,
                                    _lower_percentile_iqr_zero, _upper_percentile_iqr_zero,
                                    _max_thickness=HARD_THICKNESS_LIMIT_NM):
    top_down_data = np.max(_data, axis=0)
    top_down_data = replace_outliers_iqr(
        top_down_data,
        lower_p=_lower_percentile_iqr, upper_p=_upper_percentile_iqr,
        lower_p_zero_iqr=_lower_percentile_iqr_zero,
        upper_p_zero_iqr=_upper_percentile_iqr_zero)
    top_down_data = np.flipud(top_down_data)

    # Fixed colour-scale range (0 to the configured display max) so heatmaps
    # across samples are directly comparable: identical colour ↔ identical
    # thickness, regardless of per-sample value distribution.
    fig = go.Figure(data=go.Heatmap(
        z=top_down_data, colorscale='Viridis',
        zmin=0, zmax=_max_thickness,
        colorbar=dict(title='Thickness (nm)')))
    fig.update_layout(
        title=_title,
        # White canvas (no grey plot background) and no X/Y axis titles.
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis_scaleanchor="x", yaxis_scaleratio=1,
        xaxis=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=40, b=0))

    logging.info(f"Saving top-down view plot with aspect ratio: {_path}")
    fig.write_image(_path, width=1000, height=1000)


def save_combined_view(_data, _title, _path, _angles, _radius, _avg_thickness_per_angle,
                       _lower_percentile_iqr, _upper_percentile_iqr,
                       _lower_percentile_iqr_zero, _upper_percentile_iqr_zero,
                       _max_thickness=HARD_THICKNESS_LIMIT_NM):
    top_down_data = np.max(_data, axis=0)
    top_down_data = replace_outliers_iqr(
        top_down_data,
        lower_p=_lower_percentile_iqr, upper_p=_upper_percentile_iqr,
        lower_p_zero_iqr=_lower_percentile_iqr_zero,
        upper_p_zero_iqr=_upper_percentile_iqr_zero)
    top_down_data = np.flipud(top_down_data)

    # Fixed colour-scale range (0 to the configured display max) so heatmaps
    # across samples are directly comparable: identical colour ↔ identical
    # thickness, regardless of per-sample value distribution.
    fig = go.Figure(data=go.Heatmap(
        z=top_down_data, colorscale='Viridis',
        zmin=0, zmax=_max_thickness,
        colorbar=dict(title='Thickness (nm)')))

    center_y, center_x = top_down_data.shape[0] // 2, top_down_data.shape[1] // 2

    fig.add_shape(type="circle", xref="x", yref="y",
                  x0=center_x - _radius, y0=center_y - _radius,
                  x1=center_x + _radius, y1=center_y + _radius,
                  line_color="rgba(255,0,0,0.5)", line_width=2)

    for angle in _angles:
        fig.add_shape(
            type="line", xref="x", yref="y",
            x0=center_x, y0=center_y,
            x1=center_x + _radius * np.cos(np.deg2rad(-angle)),
            y1=center_y + _radius * np.sin(np.deg2rad(-angle)),
            line_color="rgba(128,128,128,0.3)", line_width=1)

    # `_avg_thickness_per_angle` may contain NaN for angle bins with no
    # GBM data; skip those when normalising and annotating so empty bins
    # don't get plotted as a spike at 0 nm (the previous behaviour).
    thickness_arr = np.asarray(_avg_thickness_per_angle, dtype=float)
    max_thickness = np.nanmax(thickness_arr) if np.any(~np.isnan(thickness_arr)) else 1.0
    if max_thickness == 0 or np.isnan(max_thickness):
        max_thickness = 1.0
    normalized_thickness = (thickness_arr / max_thickness) * _radius
    valid = ~np.isnan(thickness_arr)
    x_coords = center_x + normalized_thickness * np.cos(np.deg2rad(-_angles))
    y_coords = center_y + normalized_thickness * np.sin(np.deg2rad(-_angles))
    fig.add_trace(go.Scatter(
        x=x_coords[valid], y=y_coords[valid], mode='lines',
        line=dict(color='white', width=2)))

    for i, angle in enumerate(_angles[:-1]):
        avg_thickness = thickness_arr[i]
        if np.isnan(avg_thickness):
            continue  # don't annotate empty bins
        text_radius = _radius * 0.9
        text_x = center_x + text_radius * np.cos(np.deg2rad(-angle))
        text_y = center_y + text_radius * np.sin(np.deg2rad(-angle))
        fig.add_annotation(
            x=text_x, y=text_y,
            text=f"<b>{avg_thickness:.0f} nm</b>",
            showarrow=False,
            font=dict(size=12, color="white"),
            textangle=angle, xanchor="center", yanchor="middle")

    fig.update_layout(
        title=_title,
        # White canvas (no grey plot background) and no X/Y axis titles.
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis_scaleanchor="x", yaxis_scaleratio=1,
        xaxis=dict(title=None, showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title=None, showticklabels=False, showgrid=False, zeroline=False))

    logging.info(f"Saving combined view plot: {_path}")
    fig.write_image(_path, width=1000, height=1000)


def cliffs_delta(x, y):
    """Cliff's delta — non-parametric effect size between two samples.
    Returns a value in [-1, +1]. Interpretation (Romano et al. 2006):
    |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, ≥ 0.474 large.

    O(n×m) implementation — fine for the sample sizes we have (per-sample
    means / O(10-100) values per group).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    n = x.size * y.size
    diffs = x[:, None] - y[None, :]
    gt = int(np.sum(diffs > 0))
    lt = int(np.sum(diffs < 0))
    return (gt - lt) / n


def _cliffs_magnitude(d):
    """Human-readable magnitude label for Cliff's delta."""
    if np.isnan(d):
        return "n/a"
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# =========================================================================
# Publication-quality figures
#
# These are ADDITIVE: the existing per-sample PNGs (histograms, polar,
# top-down, combined view, comparative box plot) are still written exactly
# as before. The functions below write a curated, consistently-styled set
# of figures into a `publication/` subdirectory, in vector (PDF/SVG) +
# raster (PNG) formats, suitable for direct inclusion in a manuscript.
# =========================================================================

def group_color(_group: str) -> str:
    """Stable colour for a disease group (colourblind-safe)."""
    return GROUP_COLORS.get(_group, GROUP_COLORS["Unknown"])


def _save_fig_multi(_fig, _base_path: Path, _width=1200, _height=900,
                    _formats=PUB_FIGURE_FORMATS):
    """Write a figure to PNG + PDF + SVG (whichever the renderer supports).
    Each format is attempted independently so a missing engine for one
    doesn't abort the others. `_base_path` is the path WITHOUT extension.
    """
    written = []
    for fmt in _formats:
        out = _base_path.with_suffix(f".{fmt}")
        # Up-scale only raster formats; PDF/SVG are vector (scale is a no-op
        # there and can distort the page size, so leave them at 1).
        scale = PUB_PNG_SCALE if fmt in ("png", "jpg", "jpeg", "webp") else 1
        try:
            _fig.write_image(str(out), width=_width, height=_height,
                             format=fmt, scale=scale)
            written.append(str(out))
        except Exception as e:  # noqa: BLE001 — keep going on a bad engine
            logging.warning("Could not write %s (%s): %s", out, fmt, e)
    if written:
        logging.info("Saved publication figure: %s", ", ".join(written))
    return written


def _subsample(_arr, _n=20000, _seed=88233474):
    """Deterministically subsample a 1-D array to at most `_n` points.
    Used to keep violin / QQ figures light when a sample has millions of
    voxels — the distribution shape is preserved, the file size isn't
    blown up. Seeded for reproducibility (same seed as the training RNG)."""
    arr = np.asarray(_arr, dtype=float)
    if arr.size <= _n:
        return arr
    rng = np.random.default_rng(_seed)
    idx = rng.choice(arr.size, size=_n, replace=False)
    return arr[idx]


def compute_normality_report(_samples_data, _sample_names, _max_shapiro=5000):
    """Per-sample distribution diagnostics that validate the lognormal
    assumption behind the left-censored MLE correction.

    For each sample we test the RAW thickness and the LOG-transformed
    thickness for normality (Shapiro-Wilk), and report skewness + excess
    kurtosis. If the log-transformed data is "more normal" than the raw
    data (higher Shapiro W), the lognormal model is the better fit — which
    is the assumption the censored MLE relies on.

    Shapiro-Wilk is capped at N=5000 (its valid range); larger samples are
    deterministically subsampled.

    Returns a dict suitable for yaml serialisation.
    """
    report = {
        "method": (
            "Shapiro-Wilk normality test on raw vs log-transformed "
            "thickness (subsampled to N<=5000), plus skewness and excess "
            "kurtosis. Log-normality is supported when the log-transformed "
            "data has the higher Shapiro W statistic. Computed on the "
            "less-processed data (post hard-cap + COL4 mask, BEFORE IQR "
            "outlier removal) so tail-clipping does not artificially "
            "depress the normality statistic. The QQ grid is the visual "
            "arbiter — read it for samples where lognormal_preferred is "
            "False, since truncation/censoring can still distort the test."),
        "samples": [],
    }
    n_supports_lognormal = 0
    n_total = 0
    for data, name in zip(_samples_data, _sample_names):
        data = np.asarray(data, dtype=float)
        data = data[data > 0]
        if data.size < 8:
            report["samples"].append({
                "sample_name": name, "n": int(data.size),
                "note": "too few positive voxels for normality testing"})
            continue
        sub = _subsample(data, _max_shapiro)
        log_sub = np.log(sub)
        try:
            w_raw, p_raw = shapiro(sub)
            w_log, p_log = shapiro(log_sub)
        except Exception as e:  # noqa: BLE001
            report["samples"].append({
                "sample_name": name, "n": int(data.size),
                "note": f"shapiro failed: {e}"})
            continue
        lognormal_preferred = bool(w_log > w_raw)
        n_total += 1
        if lognormal_preferred:
            n_supports_lognormal += 1
        report["samples"].append({
            "sample_name": name,
            "n": int(data.size),
            "n_tested": int(sub.size),
            "shapiro_W_raw": float(w_raw),
            "shapiro_p_raw": float(p_raw),
            "shapiro_W_log": float(w_log),
            "shapiro_p_log": float(p_log),
            "skewness_raw": float(skew(sub)),
            "excess_kurtosis_raw": float(kurtosis(sub)),
            "skewness_log": float(skew(log_sub)),
            "excess_kurtosis_log": float(kurtosis(log_sub)),
            "lognormal_preferred": lognormal_preferred,
        })
    if n_total > 0:
        report["summary"] = {
            "n_samples_tested": n_total,
            "n_lognormal_preferred": n_supports_lognormal,
            "fraction_lognormal_preferred": float(n_supports_lognormal / n_total),
            "interpretation": (
                "Fraction of samples where the log-transform is closer to "
                "normal than the raw values. Close to 1.0 supports the "
                "lognormal model used by the censored-MLE thickness "
                "correction."),
        }
    return report


def _shared_ymax(_samples_data, _hi_percentile=99.0, _pad=1.10, _floor=100.0):
    """Compute a shared y-axis upper limit for the thickness figures.

    Uses the max across samples of a high percentile (default P99) so a
    single extreme voxel doesn't stretch the axis, padded by `_pad` and
    rounded up to a clean multiple of 50. Capped at HARD_THICKNESS_LIMIT_NM.
    A data-driven range (rather than a fixed 0–1200) makes the data fill
    the panel — figures within one stats run all share this same limit so
    they remain mutually comparable.
    """
    highs = []
    for d in _samples_data:
        d = np.asarray(d, dtype=float)
        d = d[d > 0]
        if d.size:
            highs.append(float(np.percentile(d, _hi_percentile)))
    if not highs:
        return _floor
    ymax = max(highs) * _pad
    ymax = float(np.ceil(ymax / 50.0) * 50.0)
    return float(min(max(ymax, _floor), HARD_THICKNESS_LIMIT_NM))


def _p_to_stars(_p) -> str:
    """Significance-star notation for a p-value."""
    if _p is None or np.isnan(_p):
        return "n/a"
    if _p < 0.001:
        return "***"
    if _p < 0.01:
        return "**"
    if _p < 0.05:
        return "*"
    return "ns"


# =========================================================================
# Group-comparison figures (nested-data aware)
#
# These compare disease GROUPS, with the statistical unit made explicit.
# The morph pipeline produces many images (Series) per mouse; images from
# one mouse are pseudo-replicates of a single animal, so the genuine unit
# is the MOUSE. Each figure here is generated once per unit ("image" and
# "mouse") into its own directory, so the per-image and per-mouse analyses
# sit side by side and the reader can see how the conclusion depends on the
# unit. With only a few animals per group the p-values are underpowered, so
# these figures foreground EFFECT SIZE + uncertainty (bootstrap CIs) over
# significance stars.
# =========================================================================

_GROUP_ORDER = ["Control", "Podocin", "Collagen"]


def _ordered_groups(_present):
    """Stable group order: the known disease groups first, then any extras
    alphabetically. `_present` is any iterable of group names."""
    present = list(dict.fromkeys(_present))
    return [g for g in _GROUP_ORDER if g in present] + \
           sorted(g for g in present if g not in _GROUP_ORDER)


def _bootstrap_mean_diff_ci(_ref, _grp, _n_boot=5000, _ci=95, _seed=88233474):
    """Bootstrap CI for mean(_grp) − mean(_ref). Returns (obs, lo, hi).

    Percentile bootstrap. Honest but wide at the small animal counts here;
    that width IS the message — it tells the reader how little the data
    constrains the effect, which a bare p-value hides."""
    ref = np.asarray(_ref, dtype=float)
    grp = np.asarray(_grp, dtype=float)
    obs = float(np.mean(grp) - np.mean(ref)) if ref.size and grp.size else float("nan")
    if ref.size < 2 or grp.size < 2:
        return obs, float("nan"), float("nan")
    rng = np.random.default_rng(_seed)
    rb = rng.choice(ref, size=(_n_boot, ref.size), replace=True).mean(axis=1)
    gb = rng.choice(grp, size=(_n_boot, grp.size), replace=True).mean(axis=1)
    diffs = gb - rb
    a = (100 - _ci) / 2
    return obs, float(np.percentile(diffs, a)), float(np.percentile(diffs, 100 - a))


def _bootstrap_cliffs_ci(_a, _b, _n_boot=2000, _ci=95, _seed=88233474):
    """Bootstrap CI for Cliff's delta(_a, _b). Returns (obs, lo, hi)."""
    a = np.asarray(_a, dtype=float)
    b = np.asarray(_b, dtype=float)
    obs = cliffs_delta(a, b)
    if a.size < 2 or b.size < 2:
        return obs, float("nan"), float("nan")
    rng = np.random.default_rng(_seed)
    vals = np.empty(_n_boot)
    for i in range(_n_boot):
        vals[i] = cliffs_delta(rng.choice(a, a.size, replace=True),
                               rng.choice(b, b.size, replace=True))
    lo = (100 - _ci) / 2
    return obs, float(np.percentile(vals, lo)), float(np.percentile(vals, 100 - lo))


def _sig_brackets(_fig, _significance, _pos, _ymax, _y0_frac=0.82, _step_frac=0.09):
    """Draw Bonferroni-Mann-Whitney significance brackets onto a numeric-x
    figure for the pairs flagged significant_05. `_pos` maps group→x."""
    if not (_significance and _significance.get("pairwise")):
        return
    sig_pairs = [p for p in _significance["pairwise"]
                 if p.get("significant_05") and p["a"] in _pos and p["b"] in _pos]
    y0, step = _ymax * _y0_frac, _ymax * _step_frac
    for k, p in enumerate(sig_pairs):
        xa, xb = _pos[p["a"]], _pos[p["b"]]
        y = y0 + k * step
        _fig.add_shape(type="line", x0=xa, x1=xb, y0=y, y1=y,
                       line=dict(color="black", width=1.5))
        for xx in (xa, xb):
            _fig.add_shape(type="line", x0=xx, x1=xx, y0=y - step * 0.18, y1=y,
                           line=dict(color="black", width=1.5))
        _fig.add_annotation(x=(xa + xb) / 2, y=y + step * 0.12,
                            text=_p_to_stars(p["p_bonferroni"]),
                            showarrow=False, font=dict(size=PUB_ANNOT_SIZE + 2))


def _rollup_to_mouse(_per_image_records):
    """Aggregate per-image records to one record per mouse. Each input is a
    dict with keys name, mouse, group, mean, voxels. Returns a list of dicts
    with keys mouse, group, mean (mean of that mouse's image means),
    n_images, image_means, voxels (concatenated, subsampled)."""
    by_mouse = {}
    for r in _per_image_records:
        by_mouse.setdefault(r["mouse"], []).append(r)
    out = []
    for mouse, recs in by_mouse.items():
        image_means = [float(r["mean"]) for r in recs]
        vox = np.concatenate([_subsample(np.asarray(r["voxels"], dtype=float),
                                          10000) for r in recs]) \
            if recs else np.array([])
        out.append({
            "mouse": mouse, "group": recs[0]["group"],
            "mean": float(np.mean(image_means)), "n_images": len(recs),
            "image_means": image_means, "voxels": vox,
        })
    return out


def _write_mouse_summary(_mouse_dir: Path, _per_mouse):
    """Write the per-animal rollup table (one row per mouse) and the
    per-group summary of mouse means as a readable YAML."""
    rows = [{"mouse": r["mouse"], "group": r["group"],
             "n_images": int(r["n_images"]),
             "mean_thickness_nm": float(r["mean"]),
             "image_means_nm": [round(float(x), 1) for x in r["image_means"]]}
            for r in sorted(_per_mouse, key=lambda x: (x["group"], x["mouse"]))]
    by_group = {}
    for r in _per_mouse:
        by_group.setdefault(r["group"], []).append(float(r["mean"]))
    group_summary = {
        g: {"n_mice": len(v),
            "mean_of_mouse_means_nm": float(np.mean(v)),
            "std_of_mouse_means_nm": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "sem_nm": (float(np.std(v, ddof=1) / np.sqrt(len(v)))
                       if len(v) > 1 else 0.0)}
        for g, v in by_group.items()}
    out = {
        "note": ("Per-mouse rollup: each mouse's value is the mean of its "
                 "image means. This is the genuine biological replicate — "
                 "use n = mice for any group comparison. The per-image "
                 "analysis under publication/by_image/ is provided for "
                 "transparency only (it overstates n via pseudoreplication)."),
        "per_mouse": rows,
        "per_group": group_summary,
    }
    with open(_mouse_dir / "mouse_summary.yaml", "w", encoding="UTF-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    logging.info("Saved per-mouse summary to %s",
                 _mouse_dir / "mouse_summary.yaml")


# =========================================================================
# Curated group-comparison figure set (the new style).
#
# Four figures, each answering a DIFFERENT question, in one publication/
# directory — no redundant box/violin/raincloud variants of the same
# summary. All statistics use the MOUSE as the replicate (per-image rolls
# up to per-mouse); figures that show raw spread also show the per-image
# cloud so the nesting is visible.
#   group_violin_comparison — the three groups side by side (main figure)
#   group_ridgeline         — stacked density ridges (distribution shape)
#   group_ecdf              — cumulative distribution (threshold crossings)
#   group_estimation        — mean difference vs Control + bootstrap CI
# =========================================================================

def _rgba(_hex, _a):
    """`#RRGGBB` + alpha → an `rgba(r,g,b,a)` string for translucent fills."""
    h = _hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{_a})"


def _theme(_fig, _title=None, _xtitle=None, _ytitle=None, _legend=False):
    """Shared editorial theme for the new figures: white canvas, soft-ink
    axes/text, centred title, no clutter. Returns the figure for chaining."""
    _fig.update_layout(
        template="simple_white",
        font=dict(family=PUB_FONT_FAMILY, size=PUB_TICK_SIZE, color=PUB_INK),
        title=dict(text=_title, font=dict(size=PUB_TITLE_SIZE, color=PUB_INK),
                   x=0.5, xanchor="center") if _title else None,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=95, r=45, t=100 if _title else 45, b=95),
        showlegend=_legend,
    )
    if _xtitle is not None:
        _fig.update_xaxes(title=dict(text=_xtitle, font=dict(size=PUB_AXIS_TITLE_SIZE)),
                          tickfont=dict(size=PUB_TICK_SIZE), showline=True,
                          linewidth=1.5, linecolor=PUB_INK, ticks="outside")
    if _ytitle is not None:
        _fig.update_yaxes(title=dict(text=_ytitle, font=dict(size=PUB_AXIS_TITLE_SIZE)),
                          tickfont=dict(size=PUB_TICK_SIZE), showline=True,
                          linewidth=1.5, linecolor=PUB_INK, ticks="outside")
    return _fig


def save_group_violin_comparison(_per_image, _per_mouse, _voxels_by_group,
                                 _base_path: Path, _significance=None, _ymax=None):
    """MAIN figure — the three disease groups side by side.

    Per group: a full violin of the pooled voxel thickness distribution, a
    thin inner quartile box with a median line, every MOUSE's mean as a
    jittered dot, and the group mean as a thick coloured crossbar. n is
    reported as mice·images, and Bonferroni-Mann-Whitney brackets (computed
    on the mouse means) sit on top. One figure carries the distribution
    shape, the per-animal spread, and the group-level test together."""
    order = _ordered_groups(g for _, _, g in _per_mouse)
    if not order:
        return
    pos = {g: i for i, g in enumerate(order)}
    if _ymax is None:
        _ymax = _shared_ymax(list(_voxels_by_group.values()))

    fig = go.Figure()
    for g in order:
        i = pos[g]
        color = group_color(g)
        vox = np.asarray(_voxels_by_group.get(g, []), dtype=float)
        vox = vox[vox > 0]
        if vox.size == 0:
            continue
        sub = _subsample(vox, 9000)
        fig.add_trace(go.Violin(
            x=np.full(sub.size, i, dtype=float), y=sub, width=0.85,
            line=dict(color=PUB_INK, width=1.2), fillcolor=_rgba(color, 0.45),
            points=False, spanmode="hard", scalemode="width",
            showlegend=False, hoverinfo="skip"))
        q1, med, q3 = np.percentile(vox, [25, 50, 75])
        fig.add_shape(type="rect", x0=i - 0.045, x1=i + 0.045, y0=q1, y1=q3,
                      line=dict(color=PUB_INK, width=1.2), fillcolor="white",
                      layer="above")
        fig.add_shape(type="line", x0=i - 0.045, x1=i + 0.045, y0=med, y1=med,
                      line=dict(color=PUB_INK, width=2.5), layer="above")
        means = [v for v, _, gg in _per_mouse if gg == g]
        rng = np.random.default_rng(88233474 + i)
        jit = (rng.random(len(means)) - 0.5) * 0.16
        fig.add_trace(go.Scatter(
            x=i + jit, y=means, mode="markers",
            marker=dict(color=color, size=13, line=dict(color=PUB_INK, width=1.5)),
            showlegend=False, hoverinfo="skip"))
        if means:
            gm = float(np.mean(means))
            fig.add_shape(type="line", x0=i - 0.22, x1=i + 0.22, y0=gm, y1=gm,
                          line=dict(color=color, width=4), layer="above")
    _theme(fig, "GBM thickness across disease groups", None, "Thickness (nm)")
    fig.update_yaxes(range=[0, _ymax])
    ticktext = []
    for g in order:
        n_mouse = len({m for _, m, gg in _per_mouse if gg == g})
        n_img = sum(1 for _, _, gg in _per_image if gg == g)
        ticktext.append(f"{g}<br>{n_mouse} mice · {n_img} img")
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=ticktext, range=[-0.6, len(order) - 0.4],
                     showline=True, linewidth=1.5, linecolor=PUB_INK,
                     ticks="outside", tickfont=dict(size=PUB_TICK_SIZE))
    _sig_brackets(fig, _significance, pos, _ymax)
    fig.add_annotation(
        x=0, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
        text="violin = voxel distribution · box = IQR · dots = per-mouse means "
             "· bar = group mean", showarrow=False,
        font=dict(size=PUB_ANNOT_SIZE - 2, color=PUB_INK))
    _save_fig_multi(fig, _base_path, _width=max(840, 320 * len(order)),
                    _height=900)


def save_group_ridgeline(_voxels_by_group, _base_path: Path, _xmax=None):
    """Stacked density ridges, one per group (a 'joyplot'). A visually
    distinct read of the *shape* of each group's thickness distribution and
    how it shifts/broadens between groups — answers 'is the change a uniform
    shift or a heavy tail?' which a mean or box cannot. A dotted line marks
    each group's median."""
    order = _ordered_groups(_voxels_by_group.keys())
    if not order:
        return
    if _xmax is None:
        _xmax = _shared_ymax(list(_voxels_by_group.values()))
    grid = np.linspace(0, _xmax, 400)
    # `scale` is the peak ridge height in units of the row `spacing`. Keep it
    # below 1.0 so a ridge's peak stays under the next baseline — only the
    # lower tails overlap slightly (classic ridgeline look without the clutter
    # of the earlier 1.7 over-scaling).
    scale, spacing = 0.9, 1.0
    # Control on top, so baselines descend down the figure.
    baseline = {g: (len(order) - 1 - idx) * spacing for idx, g in enumerate(order)}

    fig = go.Figure()
    # Draw bottom-up so upper ridges overlap cleanly over lower ones.
    for g in sorted(order, key=lambda x: baseline[x]):
        base = baseline[g]
        color = group_color(g)
        vox = np.asarray(_voxels_by_group[g], dtype=float)
        vox = vox[(vox > 0) & (vox <= _xmax)]
        if vox.size < 10:
            continue
        dens = gaussian_kde(_subsample(vox, 20000))(grid)
        if dens.max() > 0:
            dens = dens / dens.max() * scale
        fig.add_trace(go.Scatter(x=grid, y=np.full_like(grid, base), mode="lines",
                                 line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=grid, y=base + dens, mode="lines",
            line=dict(color=PUB_INK, width=1.5), fill="tonexty",
            fillcolor=_rgba(color, 0.72), showlegend=False, hoverinfo="skip"))
        med = float(np.median(vox))
        fig.add_shape(type="line", x0=med, x1=med, y0=base, y1=base + scale * 0.55,
                      line=dict(color=PUB_INK, width=1.2, dash="dot"))
    _theme(fig, "GBM thickness distribution by group (ridgeline)",
           "Thickness (nm)", None)
    fig.update_xaxes(range=[0, _xmax])
    fig.update_yaxes(tickmode="array", tickvals=[baseline[g] for g in order],
                     ticktext=order, showline=False, ticks="", showgrid=False,
                     range=[-0.3, (len(order) - 1) * spacing + scale + 0.4])
    _save_fig_multi(fig, _base_path, _width=1100, _height=230 * len(order) + 240)


def save_group_ecdf(_voxels_by_group, _base_path: Path, _xmax=None):
    """Empirical cumulative distribution of voxel thickness, one line per
    group. Binning-free; lets the reader read off 'fraction of GBM thinner
    than X nm' and see stochastic dominance (a fully right-shifted curve =
    uniformly thicker). The dashed line marks the median crossing."""
    order = _ordered_groups(_voxels_by_group.keys())
    if not order:
        return
    if _xmax is None:
        _xmax = _shared_ymax(list(_voxels_by_group.values()))
    fig = go.Figure()
    for g in order:
        vox = np.asarray(_voxels_by_group[g], dtype=float)
        vox = np.sort(_subsample(vox[vox > 0], 30000))
        if vox.size < 2:
            continue
        y = np.arange(1, vox.size + 1) / vox.size
        fig.add_trace(go.Scatter(x=vox, y=y, mode="lines", name=g,
                                 line=dict(color=group_color(g), width=3)))
    fig.add_shape(type="line", x0=0, x1=_xmax, y0=0.5, y1=0.5,
                  line=dict(color="#BBBBBB", width=1, dash="dash"))
    _theme(fig, "Cumulative GBM thickness by group", "Thickness (nm)",
           "Cumulative fraction of voxels", _legend=True)
    fig.update_xaxes(range=[0, _xmax])
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(legend=dict(title="Group", font=dict(size=PUB_TICK_SIZE),
                                  x=0.98, xanchor="right", y=0.02,
                                  yanchor="bottom"))
    _save_fig_multi(fig, _base_path, _width=1100, _height=850)


def save_group_estimation(_values_by_group, _base_path: Path,
                          _ref_group="Control", _unit_label="mouse"):
    """Cumming shared-control estimation plot. Top: each unit's value (one
    dot per mouse) per group + the group mean. Bottom: the mean difference
    of every group vs the reference with a bootstrap 95% CI. Effect size +
    uncertainty instead of a yes/no p-value — the honest summary when only a
    few animals per group are available."""
    order = _ordered_groups(_values_by_group.keys())
    if len(order) < 2:
        return
    ref = _ref_group if _ref_group in order else order[0]
    pos = {g: i for i, g in enumerate(order)}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.06)
    rng = np.random.default_rng(88233474)
    for g in order:
        vals = np.asarray(_values_by_group[g], dtype=float)
        if vals.size == 0:
            continue
        jit = (rng.random(vals.size) - 0.5) * 0.16
        fig.add_trace(go.Scatter(
            x=pos[g] + jit, y=vals, mode="markers",
            marker=dict(color=group_color(g), size=11,
                        line=dict(color=PUB_INK, width=1)),
            showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[pos[g] - 0.2, pos[g] + 0.2], y=[vals.mean()] * 2, mode="lines",
            line=dict(color=PUB_INK, width=3), showlegend=False,
            hoverinfo="skip"), row=1, col=1)
    fig.add_shape(type="line", x0=-0.5, x1=len(order) - 0.5, y0=0, y1=0,
                  line=dict(color="#999999", width=1, dash="dash"), row=2, col=1)
    for g in order:
        if g == ref:
            continue
        obs, lo, hi = _bootstrap_mean_diff_ci(_values_by_group[ref],
                                              _values_by_group[g])
        ep = hi - obs if np.isfinite(hi) else 0
        em = obs - lo if np.isfinite(lo) else 0
        fig.add_trace(go.Scatter(
            x=[pos[g]], y=[obs], mode="markers",
            marker=dict(color=group_color(g), size=15,
                        line=dict(color=PUB_INK, width=1.5)),
            error_y=dict(type="data", symmetric=False, array=[ep],
                         arrayminus=[em], thickness=2, width=9, color=PUB_INK),
            showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.update_layout(
        template="simple_white",
        font=dict(family=PUB_FONT_FAMILY, size=PUB_TICK_SIZE, color=PUB_INK),
        title=dict(text=(f"GBM thickness — estimation plot (unit = {_unit_label})"
                         f"<br><sub>top: per-{_unit_label} values + mean; bottom: "
                         f"mean difference vs {ref} ± bootstrap 95% CI</sub>"),
                   font=dict(size=PUB_TITLE_SIZE, color=PUB_INK), x=0.5,
                   xanchor="center"),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=95, r=45, t=115, b=70))
    fig.update_yaxes(title=dict(text="Thickness (nm)",
                                font=dict(size=PUB_AXIS_TITLE_SIZE)),
                     showline=True, linecolor=PUB_INK, ticks="outside",
                     row=1, col=1)
    fig.update_yaxes(title=dict(text=f"Δ vs {ref} (nm)",
                                font=dict(size=PUB_AXIS_TITLE_SIZE)),
                     showline=True, linecolor=PUB_INK, ticks="outside",
                     row=2, col=1)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=order, range=[-0.5, len(order) - 0.5],
                     showline=True, linecolor=PUB_INK, ticks="outside",
                     tickfont=dict(size=PUB_TICK_SIZE), row=2, col=1)
    _save_fig_multi(fig, _base_path, _width=max(840, 300 * len(order)),
                    _height=900)


def save_mouse_violin_comparison(_per_mouse, _base_path: Path, _ymax=None):
    """Per-MOUSE distribution plot: one violin per animal (the pooled voxel
    thickness of that mouse's images), ordered and coloured by disease group,
    each with the mouse-mean diamond + inner IQR box. Shows the
    mouse-to-mouse variability that the group figure aggregates over — the
    per-animal view of the same data. Group labels sit above each block and
    dotted lines separate the groups."""
    order = _ordered_groups(r['group'] for r in _per_mouse)
    if not order:
        return
    rows = sorted(_per_mouse,
                  key=lambda r: (order.index(r['group']), str(r['mouse'])))
    if _ymax is None:
        _ymax = _shared_ymax([r['voxels'] for r in rows])

    fig = go.Figure()
    for i, r in enumerate(rows):
        color = group_color(r['group'])
        vox = np.asarray(r['voxels'], dtype=float)
        vox = vox[vox > 0]
        if vox.size == 0:
            continue
        sub = _subsample(vox, 9000)
        fig.add_trace(go.Violin(
            x=np.full(sub.size, i, dtype=float), y=sub, width=0.85,
            line=dict(color=PUB_INK, width=1), fillcolor=_rgba(color, 0.45),
            points=False, spanmode="hard", scalemode="width",
            showlegend=False, hoverinfo="skip"))
        q1, med, q3 = np.percentile(vox, [25, 50, 75])
        fig.add_shape(type="rect", x0=i - 0.05, x1=i + 0.05, y0=q1, y1=q3,
                      line=dict(color=PUB_INK, width=1), fillcolor="white",
                      layer="above")
        fig.add_shape(type="line", x0=i - 0.05, x1=i + 0.05, y0=med, y1=med,
                      line=dict(color=PUB_INK, width=2), layer="above")
        fig.add_trace(go.Scatter(
            x=[i], y=[r['mean']], mode="markers",
            marker=dict(color=color, size=12, symbol="diamond",
                        line=dict(color=PUB_INK, width=1.5)),
            showlegend=False, hoverinfo="skip"))
    # Dotted separators between groups + a coloured group label above each block.
    for j in range(1, len(rows)):
        if rows[j]['group'] != rows[j - 1]['group']:
            fig.add_shape(type="line", x0=j - 0.5, x1=j - 0.5, y0=0, y1=_ymax,
                          line=dict(color="#CCCCCC", width=1, dash="dot"),
                          layer="below")
    for g in order:
        idxs = [i for i, r in enumerate(rows) if r['group'] == g]
        if idxs:
            fig.add_annotation(x=(idxs[0] + idxs[-1]) / 2, y=_ymax, yref="y",
                               text=f"<b>{g}</b>", showarrow=False,
                               font=dict(size=PUB_ANNOT_SIZE, color=group_color(g)),
                               yanchor="bottom")
    _theme(fig, "GBM thickness per mouse<br><sub>one violin per animal · "
                "diamond = mouse mean · box = IQR</sub>", None, "Thickness (nm)")
    fig.update_yaxes(range=[0, _ymax])
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(rows))),
                     ticktext=[str(r['mouse']) for r in rows],
                     range=[-0.6, len(rows) - 0.4], tickangle=-40,
                     showline=True, linewidth=1.5, linecolor=PUB_INK,
                     ticks="outside", tickfont=dict(size=PUB_TICK_SIZE - 3))
    _save_fig_multi(fig, _base_path, _width=max(900, 115 * len(rows)),
                    _height=860)


def save_mouse_ecdf(_per_mouse, _base_path: Path, _xmax=None):
    """Per-MOUSE cumulative distribution: one ECDF line per animal, coloured
    by disease group. Within-group lines clustering together (vs separating
    between groups) is the visual signature of a real group effect; a stray
    line flags an outlier animal."""
    order = _ordered_groups(r['group'] for r in _per_mouse)
    if not order:
        return
    if _xmax is None:
        _xmax = _shared_ymax([r['voxels'] for r in _per_mouse])
    rows = sorted(_per_mouse,
                  key=lambda r: (order.index(r['group']), str(r['mouse'])))
    fig = go.Figure()
    for r in rows:
        vox = np.asarray(r['voxels'], dtype=float)
        vox = np.sort(_subsample(vox[vox > 0], 20000))
        if vox.size < 2:
            continue
        y = np.arange(1, vox.size + 1) / vox.size
        fig.add_trace(go.Scatter(
            x=vox, y=y, mode="lines", name=f"{r['mouse']} ({r['group'][:4]})",
            line=dict(color=group_color(r['group']), width=1.6), opacity=0.85))
    _theme(fig, "Cumulative GBM thickness per mouse", "Thickness (nm)",
           "Cumulative fraction of voxels", _legend=True)
    fig.update_xaxes(range=[0, _xmax])
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(legend=dict(title="Mouse", font=dict(size=PUB_TICK_SIZE - 5),
                                  x=0.98, xanchor="right", y=0.02,
                                  yanchor="bottom"))
    _save_fig_multi(fig, _base_path, _width=1150, _height=850)


def generate_group_figures(_stats_dir: Path, _per_image_records,
                           _max_thickness=HARD_THICKNESS_LIMIT_NM):
    """Write the curated group-comparison set into `<stats_dir>/publication/`.

    `_per_image_records` is a list of dicts (name, mouse, group, mean,
    voxels, preiqr). Produces the four figures (PNG+PDF+SVG), the per-mouse
    and per-image significance YAMLs, the per-animal rollup, and the
    lognormal normality report. Returns the figure base-names."""
    pub = _stats_dir / "publication"
    pub.mkdir(parents=True, exist_ok=True)
    if not _per_image_records:
        return []

    per_mouse = _rollup_to_mouse(_per_image_records)
    per_image_tuples = [(float(r["mean"]), r["mouse"], r["group"])
                        for r in _per_image_records]
    per_mouse_tuples = [(float(r["mean"]), r["mouse"], r["group"])
                        for r in per_mouse]
    values_by_group = {}
    for r in per_mouse:
        values_by_group.setdefault(r["group"], []).append(r["mean"])
    voxels_by_group = {}
    for r in _per_image_records:
        v = _subsample(np.asarray(r["voxels"], dtype=float), 10000)
        voxels_by_group.setdefault(r["group"], []).append(v)
    voxels_by_group = {g: np.concatenate(vs) for g, vs in voxels_by_group.items()}
    # Fixed, config-driven axis max so every figure (and both mask variants /
    # all runs) shares one thickness scale and stays directly comparable.
    ymax = _max_thickness

    # Significance at both units: mouse = the one to cite, image = transparency.
    mouse_sig = compute_group_significance(
        [{"sample_name": r["mouse"], "group": r["group"], "mean": r["mean"]}
         for r in per_mouse])
    image_sig = compute_group_significance(
        [{"sample_name": r["name"], "group": r["group"], "mean": r["mean"]}
         for r in _per_image_records])
    with open(pub / "group_significance_by_mouse.yaml", "w", encoding="UTF-8") as f:
        yaml.safe_dump(mouse_sig, f, sort_keys=False)
    with open(pub / "group_significance_by_image.yaml", "w", encoding="UTF-8") as f:
        yaml.safe_dump(image_sig, f, sort_keys=False)
    _write_mouse_summary(pub, per_mouse)
    try:
        rep = compute_normality_report(
            [r.get("preiqr", r["voxels"]) for r in _per_image_records],
            [r["name"] for r in _per_image_records])
        with open(pub / "normality_report.yaml", "w", encoding="UTF-8") as f:
            yaml.safe_dump(rep, f, sort_keys=False)
    except Exception as e:  # noqa: BLE001
        logging.warning("Normality report failed: %s", e)

    save_group_violin_comparison(per_image_tuples, per_mouse_tuples,
                                 voxels_by_group,
                                 pub / "group_violin_comparison", mouse_sig, ymax)
    save_group_ridgeline(voxels_by_group, pub / "group_ridgeline", ymax)
    save_group_ecdf(voxels_by_group, pub / "group_ecdf", ymax)
    save_group_estimation(values_by_group, pub / "group_estimation",
                          _ref_group="Control", _unit_label="mouse")
    # Per-mouse views: one violin / ECDF per animal (grouped + coloured by
    # disease group) so the mouse-to-mouse variability behind the group
    # figures is visible.
    save_mouse_violin_comparison(per_mouse, pub / "mouse_violin", ymax)
    save_mouse_ecdf(per_mouse, pub / "mouse_ecdf", ymax)
    logging.info("Wrote curated group + per-mouse figures to %s", pub)
    return sorted(p.name for p in pub.iterdir())


def compute_group_significance(_per_sample_summary):
    """Run an omnibus Kruskal-Wallis across all detected groups plus
    pairwise Mann-Whitney U tests with Bonferroni correction. Effect sizes
    (Cliff's delta) are reported alongside p-values.

    Input: a list of dicts each with 'sample_name', 'group', 'mean'.
    Returns a dict suitable for yaml serialization.
    """
    groups = {}
    for row in _per_sample_summary:
        g = row.get("group", "Unknown")
        groups.setdefault(g, []).append(float(row["mean"]))

    # Need at least two groups with ≥2 samples each to test.
    eligible = {g: vals for g, vals in groups.items() if len(vals) >= 2}
    out = {
        "method": (
            "Kruskal-Wallis omnibus across groups; pairwise Mann-Whitney U "
            "with Bonferroni correction; effect size Cliff's delta."),
        "groups": {g: {"n": len(v), "mean_of_means": float(np.mean(v)),
                       "std_of_means": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                   for g, v in groups.items()},
    }

    if len(eligible) < 2:
        out["omnibus"] = None
        out["pairwise"] = []
        out["note"] = (
            f"Insufficient groups with n≥2 for testing "
            f"(found {len(eligible)} eligible groups out of {len(groups)}).")
        return out

    # Omnibus: H_0 that all groups have the same distribution.
    h, p = kruskal(*eligible.values())
    out["omnibus"] = {
        "test": "Kruskal-Wallis", "statistic": float(h), "p_value": float(p),
        "groups_tested": list(eligible.keys()),
    }

    # Pairwise Mann-Whitney with Bonferroni correction over the number of
    # pairs tested. Two-sided.
    pairs = list(combinations(eligible.keys(), 2))
    n_pairs = max(1, len(pairs))
    pairwise = []
    for a, b in pairs:
        u, p_raw = mannwhitneyu(eligible[a], eligible[b], alternative="two-sided")
        p_corr = min(1.0, p_raw * n_pairs)
        d = cliffs_delta(eligible[a], eligible[b])
        pairwise.append({
            "a": a, "b": b,
            "a_mean": float(np.mean(eligible[a])),
            "b_mean": float(np.mean(eligible[b])),
            "U": float(u),
            "p_raw": float(p_raw),
            "p_bonferroni": float(p_corr),
            "cliffs_delta": float(d),
            "magnitude": _cliffs_magnitude(d),
            "significant_05": bool(p_corr < 0.05),
        })
    out["pairwise"] = pairwise
    return out


# Disease group of each animal, keyed by the mouse-ID token (see
# detect_mouse_id). Group is derived from the ANIMAL, not from a filename
# prefix: the old prefix table mixed "NCW.AUY380" (with prefix) and bare
# "CKM103"/"CKM110" (without), so real "NCW.CKM103…" control images never
# matched any prefix and fell through to "Unknown" — silently dropping
# control animals from the comparison. Keying off the animal token fixes
# that and keeps detect_group_type / detect_mouse_id consistent.
MOUSE_GROUPS = {
    "AUY380": "Control", "AUY381": "Control", "CKM103": "Control",
    "CKM110": "Control",
    "BDP669": "Podocin", "BDP672": "Podocin", "BDP675": "Podocin",
    "CKM104": "Collagen", "CKM105": "Collagen",
}


def detect_group_type(_sample_name):
    """Identify the disease group from the animal (mouse) ID in the filename.

    Returns ``"Unknown"`` (was previously silently ``"Control"``) when the
    name's animal token isn't in ``MOUSE_GROUPS``, and logs a warning. The
    old silent fallback would mislabel new samples as Control, contaminating
    any group-level comparison.
    """
    mouse = detect_mouse_id(_sample_name)
    group = MOUSE_GROUPS.get(mouse)
    if group is not None:
        return group
    logging.warning(
        "detect_group_type: sample '%s' (animal '%s') did not match any "
        "registered group; classifying as 'Unknown'. Add the animal to "
        "MOUSE_GROUPS in src/infer/stats.py to include this sample.",
        _sample_name, mouse)
    return "Unknown"


# Animal (mouse) ID embedded in a sample filename, e.g.
# "NCW.CKM104.20241124.lif - Series003.tiff" → "CKM104". The IDs are the
# same tokens detect_group_type keys off; several Series (images) share one
# mouse, and those images are pseudo-replicates of that ONE animal — so the
# mouse, not the image, is the genuine biological/statistical unit.
_MOUSE_ID_RE = re.compile(r"(AUY\d+|BDP\d+|CKM\d+)", re.IGNORECASE)


def detect_mouse_id(_sample_name):
    """Extract the animal (mouse) ID from a sample filename.

    Returns the matched ``AUY###`` / ``BDP###`` / ``CKM###`` token (upper
    case). Falls back to the leading filename token before the first space
    or dot, and finally ``"Unknown"`` if nothing usable is found. Used to
    aggregate the per-image measurements up to one value per animal, which
    is the correct unit for the group comparison (avoids treating multiple
    images of one mouse as independent replicates — pseudoreplication)."""
    s = str(_sample_name)
    m = _MOUSE_ID_RE.search(s.upper())
    if m:
        return m.group(1).upper()
    token = re.split(r"[ .\-_]", s.strip(), maxsplit=1)[0]
    if token:
        logging.warning(
            "detect_mouse_id: '%s' has no AUY/BDP/CKM token; falling back to "
            "leading token '%s'. Add its pattern to _MOUSE_ID_RE if this is a "
            "new animal naming scheme.", _sample_name, token)
        return token.upper()
    return "Unknown"


def fit_lognormal_left_censored(_observed, _n_censored, _censoring_threshold_nm):
    """Maximum-likelihood fit of a lognormal distribution to data with
    left-censoring at a known threshold (the PSF resolution limit).

    The morph algorithm reports thickness = 0 for surface voxels where
    measured² < PSF², i.e. the true thickness is below resolution. The
    observed (>0) voxels are a left-censored sample of the true thickness
    distribution. Averaging only the observed values overestimates the
    true mean because the lower tail of the distribution is missing.

    This function fits the parameters (mu, sigma) of the lognormal
    distribution from which the full (uncensored) data are drawn, using
    the log-likelihood:

        L(mu, sigma | data) = sum_i log f(x_i; mu, sigma)             (uncensored part)
                            + n_censored * log F(threshold; mu, sigma) (censored part)

    where f / F are the lognormal pdf / cdf. Mean of the recovered
    distribution is ``exp(mu + sigma**2 / 2)`` — the bias-corrected mean.
    A 95% CI on the mean is computed via the delta method using the
    Hessian-based covariance of the MLE.

    Returns a dict of (mu, sigma, mean_nm, median_nm, ci_lo_nm, ci_hi_nm,
    n_observed, n_censored, censoring_threshold_nm, converged). Returns
    NaN-filled values if the fit fails (too few observations, optimiser
    didn't converge, etc.) — never raises.
    """
    nan_result = {
        'mu': float('nan'), 'sigma': float('nan'),
        'mean_nm': float('nan'), 'median_nm': float('nan'),
        'ci_lo_nm': float('nan'), 'ci_hi_nm': float('nan'),
        'n_observed': int(len(_observed)),
        'n_censored': int(_n_censored),
        'censoring_threshold_nm': float(_censoring_threshold_nm),
        'converged': False,
    }
    obs = np.asarray(_observed, dtype=float)
    obs = obs[obs > 0]  # strictly positive — lognormal support
    if obs.size < 5:
        nan_result['reason'] = 'too few observed voxels'
        return nan_result

    log_obs = np.log(obs)
    log_thresh = np.log(_censoring_threshold_nm)

    def neg_log_likelihood(params):
        mu, log_sigma = params
        sigma = float(np.exp(log_sigma))
        if sigma <= 0 or not np.isfinite(sigma):
            return 1e12
        # Uncensored term: sum of lognormal log-pdf at each observation.
        # log f(x) = -log(x) - log(sigma) - 0.5*log(2*pi) - 0.5*((log(x)-mu)/sigma)**2
        ll_obs = float((-log_obs - log_sigma
                        - 0.5 * np.log(2.0 * np.pi)
                        - 0.5 * ((log_obs - mu) / sigma) ** 2).sum())
        # Censored term: n_censored × log P(X < threshold). For lognormal,
        # P(X<x) = Phi((log(x) - mu)/sigma).
        z = (log_thresh - mu) / sigma
        ll_cen = float(_n_censored) * float(norm.logcdf(z)) if _n_censored > 0 else 0.0
        total = ll_obs + ll_cen
        if not np.isfinite(total):
            return 1e12
        return -total

    # Initial estimates from the observed (positive) part of the data.
    mu0 = float(np.mean(log_obs))
    sigma0 = float(np.std(log_obs, ddof=1)) if obs.size > 1 else 0.5
    sigma0 = max(sigma0, 1e-3)

    # Nelder-Mead is derivative-free and much more robust on this 2-param
    # objective than BFGS (which often reports "precision loss" near the
    # optimum because the curvature in log-sigma is high). The objective
    # itself is well-behaved so we trust whatever minimum NM finds.
    res = minimize(neg_log_likelihood, x0=[mu0, np.log(sigma0)],
                   method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
    if not res.success:
        nan_result['reason'] = f'optimiser did not converge: {res.message}'
        return nan_result

    mu, log_sigma = res.x
    sigma = float(np.exp(log_sigma))
    mean = float(np.exp(mu + sigma ** 2 / 2.0))
    median = float(np.exp(mu))

    # Delta-method CI for the mean. Nelder-Mead doesn't return a Hessian,
    # so we estimate it numerically by second differences on the optimum.
    ci_lo = ci_hi = float('nan')
    try:
        eps = 1e-4
        f0 = neg_log_likelihood([mu, log_sigma])
        # Diagonal Hessian entries (cheaper than full numerical Hessian).
        f_mu_plus  = neg_log_likelihood([mu + eps, log_sigma])
        f_mu_minus = neg_log_likelihood([mu - eps, log_sigma])
        f_ls_plus  = neg_log_likelihood([mu, log_sigma + eps])
        f_ls_minus = neg_log_likelihood([mu, log_sigma - eps])
        h_mu_mu = (f_mu_plus + f_mu_minus - 2 * f0) / eps ** 2
        h_ls_ls = (f_ls_plus + f_ls_minus - 2 * f0) / eps ** 2
        # Off-diagonal (mu, log_sigma).
        f_pp = neg_log_likelihood([mu + eps, log_sigma + eps])
        f_pm = neg_log_likelihood([mu + eps, log_sigma - eps])
        f_mp = neg_log_likelihood([mu - eps, log_sigma + eps])
        f_mm = neg_log_likelihood([mu - eps, log_sigma - eps])
        h_off = (f_pp + f_mm - f_pm - f_mp) / (4 * eps ** 2)
        hess = np.array([[h_mu_mu, h_off], [h_off, h_ls_ls]])
        hess_inv = np.linalg.inv(hess)
        grad = np.array([mean, mean * sigma ** 2])
        var_mean = float(grad @ hess_inv @ grad)
        if var_mean > 0 and np.isfinite(var_mean):
            se_mean = float(np.sqrt(var_mean))
            ci_lo = float(mean - 1.96 * se_mean)
            ci_hi = float(mean + 1.96 * se_mean)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass

    return {
        'mu': float(mu),
        'sigma': float(sigma),
        'mean_nm': mean,
        'median_nm': median,
        'ci_lo_nm': ci_lo,
        'ci_hi_nm': ci_hi,
        'n_observed': int(obs.size),
        'n_censored': int(_n_censored),
        'censoring_threshold_nm': float(_censoring_threshold_nm),
        'converged': True,
    }


# Per-sample analysis parameters (shared by the single-process and the
# array-parallel paths).
_ALPHA_STEP = 10
_RADIUS = 1000
# Standard IQR quartiles for the box plot (was a misleading 5/95).
_Q1_PERCENTILE = 25
_Q3_PERCENTILE = 75
# Outlier-replacement endpoints for the top-down / combined heatmaps
# (colour-map saturation, not the box).
_LOWER_PCT_IQR = 25
_UPPER_PCT_IQR = 75
_LOWER_PCT_IQR_ZERO = 2
_UPPER_PCT_IQR_ZERO = 98
_BIN_SIZES = [10, 20, 50, 100]


# Each stats run is produced TWICE — once with the COL4 "negative" mask
# applied (top-percentile collagen-IV voxels zeroed: capillary lumen /
# boundary) and once without it — into separate sub-directories, so the
# effect of the mask on the thickness statistics can be compared directly.
_MASK_VARIANTS = (("with_mask", True), ("without_mask", False))


def _load_sample_for_stats(_sample_dir: Path, _inference_result_path: Path,
                           _clipping: bool, _max_thickness: int):
    """Load ONE sample's thickness volume + COL4 channel + clamp once, apply
    the (mask-independent) hard cap and optional --clipping, and return the
    pieces both mask variants share. The COL4 mask itself is NOT applied here
    — that is the per-variant step (see `_process_loaded_variant`). With
    --clipping, voxels above `_max_thickness` are dropped — the SAME value
    that bounds the figure axes, so "thickness clip" and "display max" are
    one knob. Returns a dict(data, col4_stack, group, clamp) or None."""
    clamp = None
    clamp_yaml = _sample_dir / "psf_clamp_stats.yaml"
    if clamp_yaml.exists():
        try:
            with open(clamp_yaml, encoding="UTF-8") as f:
                clamp = yaml.safe_load(f) or {}
            clamp['sample_name'] = _sample_dir.name
        except Exception as e:  # noqa: BLE001
            logging.warning("Could not read clamp stats for %s: %s",
                            _sample_dir.name, e)

    distance_file = _sample_dir / "psf_result.npz"
    if not distance_file.exists():
        logging.warning("Thickness file not found: %s", distance_file)
        return None
    try:
        data = np.load(distance_file)['arr']

        # HARD CAP — always applied, independent of --clipping or the mask.
        hard_above = int(np.sum(data > HARD_THICKNESS_LIMIT_NM))
        if hard_above > 0:
            logging.warning("Sample %s: %d voxels > %d nm hard cap — masked to 0",
                            _sample_dir.name, hard_above, HARD_THICKNESS_LIMIT_NM)
            data[data > HARD_THICKNESS_LIMIT_NM] = 0

        if _clipping:
            max_clip = min(_max_thickness, HARD_THICKNESS_LIMIT_NM)
            logging.info("Clipping enabled, removing values above %d", max_clip)
            before = np.count_nonzero(data)
            data[data > max_clip] = 0
            altered = before - np.count_nonzero(data)
            if before > 0:
                logging.info("Altered %d voxels (%.2f%% of non-zero).",
                             altered, altered / before * 100)

        sample = tifffile.imread(
            _inference_result_path / _sample_dir.name / "prediction.tif")
        col4_stack = sample[:, 1, :, :]
        sample_group = detect_group_type(_sample_dir.name)
        return {'data': data, 'col4_stack': col4_stack,
                'group': sample_group, 'clamp': clamp}
    except Exception as e:  # noqa: BLE001
        logging.error("Error loading %s: %s", distance_file, e)
        return None


def _process_loaded_variant(_sample_name: str, _variant_stats_dir: Path,
                            _loaded: dict, _apply_mask: bool,
                            _max_thickness=HARD_THICKNESS_LIMIT_NM):
    """Run the per-sample stats for ONE mask variant from preloaded data.

    Optionally applies the COL4 mask (when `_apply_mask`), then IQR outlier
    removal + summary + censored MLE, and writes the per-sample plots into
    ``<variant_stats_dir>/<sample>/``. Returns the result dict or None. The
    input volume is copied so the two variants never interfere."""
    sample_hist_dir = _variant_stats_dir / _sample_name
    sample_hist_dir.mkdir(parents=True, exist_ok=True)
    tag = "masked" if _apply_mask else "unmasked"
    try:
        data = _loaded['data'].copy()
        sample_group = _loaded['group']
        clamp = _loaded['clamp']

        if _apply_mask:
            col4_stack = _loaded['col4_stack']
            # Uniform threshold across all groups (was 91/92.5/97 per group).
            mask_threshold = np.percentile(col4_stack, UNIFORM_MASK_PERCENTILE)
            mask = (col4_stack > mask_threshold).astype(np.uint8)
            # 3D opening (was per-Z 2D, inconsistent across the Z-upsample).
            mask = binary_opening(mask,
                                  structure=np.ones((1, 3, 3))).astype(np.uint8)
            data[mask > 0] = 0

        data_no_zeros = data[data != 0]
        if len(data_no_zeros) == 0:
            logging.warning("Sample %s (%s): no non-zero voxels", _sample_name, tag)
            return None

        # Log-scale Tukey fences — thickness is right-skewed, so a raw-scale
        # symmetric fence would clip the thick-membrane tail and bias the
        # mean down. data_clean is the descriptive box / violin / per-sample
        # mean; the censored MLE below is fit on the LESS-processed data.
        data_clean = remove_outliers_iqr(data_no_zeros, _log=True)
        if len(data_clean) == 0:
            logging.warning("Sample %s (%s): no voxels left after outlier removal",
                            _sample_name, tag)
            return None

        q1 = np.percentile(data_clean, _Q1_PERCENTILE)
        median = np.median(data_clean)
        q3 = np.percentile(data_clean, _Q3_PERCENTILE)
        iqr = q3 - q1
        lowerfence = max(np.min(data_clean), q1 - 1.5 * iqr)
        upperfence = min(np.max(data_clean), q3 + 1.5 * iqr)
        mean = float(np.mean(data_clean))
        std = float(np.std(data_clean, ddof=1)) if len(data_clean) > 1 else 0.0

        # Bias-corrected mean via left-censored lognormal MLE, fit on the
        # pre-IQR data (only PSF-censoring is modelled).
        if clamp is not None:
            n_cens = int(clamp.get('clamp_count', 0))
            psf_lat = float(clamp.get('psf_lateral_nm', 149))
        else:
            n_cens, psf_lat = 0, 149.0
        censored_fit = fit_lognormal_left_censored(data_no_zeros, n_cens, psf_lat)

        summary = {
            'sample_name': _sample_name, 'group': sample_group,
            'q1': float(q1), 'median': float(median), 'q3': float(q3),
            'lowerfence': float(lowerfence), 'upperfence': float(upperfence),
            'mean': mean, 'std': std,
            'censored_mean': float(censored_fit['mean_nm']),
            'censored_median': float(censored_fit['median_nm']),
            'censored_ci_lo': float(censored_fit['ci_lo_nm']),
            'censored_ci_hi': float(censored_fit['ci_hi_nm']),
            'censored_converged': bool(censored_fit['converged']),
        }

        for bins in _BIN_SIZES:
            save_histogram(
                data_clean,
                f'Thickness Histogram with {bins} Bins - {_sample_name}',
                sample_hist_dir / f'thickness_{_sample_name}_{bins}_bins.png',
                bins)
        angles, avg_thickness = calculate_cylindrical_analysis(
            data, _ALPHA_STEP, _RADIUS)
        save_polar_plot(
            angles, avg_thickness, f'Cylindrical Analysis - {_sample_name}',
            sample_hist_dir / f'cylindrical_analysis_{_sample_name}.png',
            _max_thickness)
        save_top_down_view_aspect_ratio(
            data, f'Top-Down View (Aspect Ratio) - {_sample_name}',
            sample_hist_dir / f'top_down_view_aspect_ratio_{_sample_name}.png',
            _LOWER_PCT_IQR, _UPPER_PCT_IQR, _LOWER_PCT_IQR_ZERO, _UPPER_PCT_IQR_ZERO,
            _max_thickness)
        save_combined_view(
            data, f'Combined View - {_sample_name}',
            sample_hist_dir / f'combined_view_{_sample_name}.png',
            angles, _RADIUS, avg_thickness,
            _LOWER_PCT_IQR, _UPPER_PCT_IQR, _LOWER_PCT_IQR_ZERO, _UPPER_PCT_IQR_ZERO,
            _max_thickness)

        return {
            'name': _sample_name, 'group': sample_group,
            'summary': summary, 'clamp': clamp,
            'data_clean': data_clean, 'data_preiqr': data_no_zeros,
        }
    except Exception as e:  # noqa: BLE001
        logging.error("Error processing %s (%s): %s", _sample_name, tag, e)
        return None


def calculate_stats(_inference_result_path: Path,
                    _stats_dir: Path,
                    _clipping: bool,
                    _max_thickness: int = HARD_THICKNESS_LIMIT_NM):
    """Single-process stats: process every sample serially for BOTH mask
    variants (loaded once each), then reduce each variant into its own
    sub-directory <stats_dir>/{with_mask,without_mask}/. For the parallel
    path see calculate_stats_one_sample + _reduce. `_max_thickness` is the
    single configurable value every figure axis / colour-scale follows AND
    the --clipping drop threshold."""
    logging.info("Starting statistical analysis for inference results")
    logging.info("Input path: %s", _inference_result_path)
    logging.info("Output path: %s (with_mask/ + without_mask/)", _stats_dir)

    _stats_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = [d for d in _inference_result_path.iterdir() if d.is_dir()]
    total_samples = len(sample_dirs)
    logging.info("Found %d samples to process", total_samples)

    results = {vname: [] for vname, _ in _MASK_VARIANTS}
    for i, sample_dir in enumerate(sample_dirs, start=1):
        logging.info("Processing sample %s (%d/%d)",
                     sample_dir.name, i, total_samples)
        loaded = _load_sample_for_stats(sample_dir, _inference_result_path,
                                        _clipping, _max_thickness)
        if loaded is None:
            continue
        for vname, apply_mask in _MASK_VARIANTS:
            r = _process_loaded_variant(sample_dir.name, _stats_dir / vname,
                                        loaded, apply_mask, _max_thickness)
            if r is not None:
                results[vname].append(r)

    for vname, apply_mask in _MASK_VARIANTS:
        _reduce_and_write(_stats_dir / vname, _inference_result_path,
                          results[vname], total_samples, _mask_applied=apply_mask,
                          _max_thickness=_max_thickness)


def _reduce_and_write(_stats_dir: Path, _inference_result_path: Path,
                      _results, _total_samples, _mask_applied=True,
                      _max_thickness=HARD_THICKNESS_LIMIT_NM):
    """Aggregate the per-sample results into the cohort-level outputs:
    summary_statistics.npz, the voxel- and sample-weighted aggregates,
    group significance, the publication figure set, and metadata.txt.
    Shared by the single-process and array paths. `_mask_applied` records
    which COL4-mask variant this directory holds (for metadata);
    `_max_thickness` is the configurable display max for the figures."""
    summary_data_list = [r['summary'] for r in _results]
    all_thickness_data_for_violin = [r['data_clean'] for r in _results]
    all_thickness_data_before_outliers = [r['data_preiqr'] for r in _results]
    clamp_stats_list = [r['clamp'] for r in _results if r['clamp'] is not None]
    bin_sizes = _BIN_SIZES
    _alpha_step = _ALPHA_STEP
    _radius = _RADIUS
    _q1_percentile = _Q1_PERCENTILE
    _q3_percentile = _Q3_PERCENTILE

    # Annotate (do NOT exclude) samples with high PSF-clamp activation.
    # The previous behaviour silently dropped these from aggregation, but
    # for thin pathological tissue the clamp activation IS the signal —
    # excluding it removes exactly the samples we want to study. Keep the
    # whole cohort in the aggregate / significance, and flag the clamp
    # bias alongside each sample's row so readers can adjust expectations.
    clamp_by_name = {row['sample_name']: float(row.get('clamp_percentage', 0.0))
                     for row in clamp_stats_list}
    flagged_high_clamp = []
    for row in summary_data_list:
        pct = clamp_by_name.get(row['sample_name'])
        if pct is not None and pct > PSF_CLAMP_WARNING_PCT:
            logging.warning(
                "Sample %s: PSF clamp = %.2f%% (> %.2f%% warning threshold) — "
                "mean thickness is biased upward",
                row['sample_name'], pct, PSF_CLAMP_WARNING_PCT)
            flagged_high_clamp.append({'sample_name': row['sample_name'],
                                       'clamp_percentage': pct})
    # The full sample list is used for aggregates + significance; the
    # `flagged_high_clamp` list is purely diagnostic for metadata.txt.
    summary_clean = summary_data_list

    if summary_data_list:
        dtype = [('sample_name', 'U100'), ('group', 'U32'),
                 ('q1', 'f8'), ('median', 'f8'), ('q3', 'f8'),
                 ('lowerfence', 'f8'), ('upperfence', 'f8'),
                 ('mean', 'f8'), ('std', 'f8'),
                 # Bias-corrected (left-censored lognormal MLE) fields.
                 ('censored_mean', 'f8'), ('censored_median', 'f8'),
                 ('censored_ci_lo', 'f8'), ('censored_ci_hi', 'f8'),
                 ('censored_converged', '?')]
        # Use a fixed key order so the structured array fields line up with
        # `dtype` regardless of Python dict insertion order quirks.
        field_order = ['sample_name', 'group', 'q1', 'median', 'q3',
                       'lowerfence', 'upperfence', 'mean', 'std',
                       'censored_mean', 'censored_median',
                       'censored_ci_lo', 'censored_ci_hi',
                       'censored_converged']
        records = [tuple(d[k] for k in field_order) for d in summary_data_list]
        summary_array = np.array(records, dtype=dtype)
        summary_file = _stats_dir / "summary_statistics.npz"
        np.savez_compressed(summary_file, arr=summary_array)
        logging.info("Saved summary statistics to %s", summary_file)

    if all_thickness_data_before_outliers:
        # Voxel-weighted aggregate (concatenation of all voxels). Larger
        # samples contribute proportionally more values.
        aggregated_thickness_raw = np.concatenate(all_thickness_data_before_outliers)
        logging.info(
            "Aggregated %d raw thickness values from %d samples (before outlier removal)",
            len(aggregated_thickness_raw), len(all_thickness_data_before_outliers))
        raw_thickness_file = _stats_dir / "aggregated_thickness_data.npz"
        np.savez_compressed(raw_thickness_file, arr=aggregated_thickness_raw)
        logging.info("Saved aggregated raw thickness data to %s", raw_thickness_file)

    # Sample-weighted aggregate — equal weight per sample. This is the
    # appropriate aggregation for biological / clinical comparisons where
    # each sample is one biological unit, not one voxel. Both the raw and
    # the bias-corrected (left-censored lognormal MLE) means are
    # aggregated; the censored version is what to cite as the unbiased
    # cohort-mean estimate when PSF clamping is non-negligible.
    if summary_clean:
        sample_means = np.array([float(d['mean']) for d in summary_clean])
        cens_means = np.array([float(d['censored_mean'])
                                for d in summary_clean
                                if np.isfinite(d.get('censored_mean', float('nan')))])
        sample_weighted = {
            "n_samples": int(len(sample_means)),
            "raw": {
                "mean_of_sample_means": float(np.mean(sample_means)),
                "std_of_sample_means": float(np.std(sample_means, ddof=1)) if len(sample_means) > 1 else 0.0,
                "sem": (float(np.std(sample_means, ddof=1) / np.sqrt(len(sample_means)))
                        if len(sample_means) > 1 else 0.0),
                "median_of_sample_means": float(np.median(sample_means)),
                "note": ("Raw mean — biased upward when PSF clamping is "
                         "non-trivial. Use the `censored` block for "
                         "publication-grade cohort means."),
            },
        }
        if cens_means.size > 0:
            sample_weighted["censored"] = {
                "n_samples": int(cens_means.size),
                "mean_of_sample_means": float(np.mean(cens_means)),
                "std_of_sample_means": float(np.std(cens_means, ddof=1)) if cens_means.size > 1 else 0.0,
                "sem": (float(np.std(cens_means, ddof=1) / np.sqrt(cens_means.size))
                        if cens_means.size > 1 else 0.0),
                "median_of_sample_means": float(np.median(cens_means)),
                "method": ("Per-sample lognormal MLE with left-censoring "
                           "at the lateral PSF resolution (149 nm); the "
                           "cohort-level mean is the mean of those per-"
                           "sample bias-corrected means."),
            }
        sw_path = _stats_dir / "sample_weighted_aggregate.yaml"
        with open(sw_path, "w", encoding="UTF-8") as f:
            yaml.safe_dump(sample_weighted, f, sort_keys=False)
        logging.info("Saved sample-weighted aggregate to %s", sw_path)

    # Group-level significance testing (Kruskal-Wallis + pairwise
    # Mann-Whitney U with Bonferroni correction). Uses the clamp-filtered
    # summary so high-clamp samples don't bias the test.
    sig = None
    if summary_clean:
        sig = compute_group_significance(summary_clean)
        sig_path = _stats_dir / "group_significance.yaml"
        with open(sig_path, "w", encoding="UTF-8") as f:
            yaml.safe_dump(sig, f, sort_keys=False)
        logging.info("Saved group significance results to %s", sig_path)

    # Curated group-comparison figure set (the only figures now). One
    # `publication/` directory, one figure per question — no redundant
    # box/violin/raincloud variants of the same summary. Statistics use the
    # MOUSE as the replicate (per-image rolls up to per-mouse); the violin
    # also shows the per-image cloud so the nesting is visible. See
    # generate_group_figures for the full set + the separate per-image /
    # per-mouse significance records.
    per_image_records = [
        {"name": r["name"], "mouse": detect_mouse_id(r["name"]),
         "group": r["group"], "mean": float(r["summary"]["mean"]),
         "voxels": r["data_clean"],
         "preiqr": r["data_preiqr"]}
        for r in _results]
    if per_image_records:
        try:
            generate_group_figures(_stats_dir, per_image_records, _max_thickness)
        except Exception as e:  # noqa: BLE001 — never let figures abort stats
            logging.error("Group figure generation failed: %s", e)

    metadata_file = _stats_dir / "metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Statistical Analysis Metadata\n")
        f.write("============================\n")
        f.write(f"Input directory: {_inference_result_path}\n")
        f.write(f"Output directory: {_stats_dir}\n")
        f.write(f"Total samples processed: {_total_samples}\n")
        f.write(f"Samples with valid data: {len(summary_data_list)}\n")
        f.write(f"Samples included in aggregate / significance: {len(summary_clean)}\n")
        f.write(f"Samples flagged (PSF clamp > {PSF_CLAMP_WARNING_PCT}%): "
                f"{len(flagged_high_clamp)}  (flagged != excluded — they "
                f"are kept in the aggregate; the flag warns that the mean "
                f"thickness is biased upward.)\n")
        f.write(f"Bin sizes used: {bin_sizes}\n")
        f.write(f"Alpha step for cylindrical analysis: {_alpha_step}\n")
        f.write(f"Radius for cylindrical analysis: {_radius}\n")
        f.write(f"Hard thickness cap (nm): {HARD_THICKNESS_LIMIT_NM}\n")
        f.write(f"COL4 negative mask: {'ON' if _mask_applied else 'OFF'} "
                f"(this is the '{'with_mask' if _mask_applied else 'without_mask'}' "
                f"variant; the masked variant zeros the top "
                f"{UNIFORM_MASK_PERCENTILE:.0f}th-percentile COL4 voxels — "
                f"capillary lumen / boundary — before computing thickness)\n")
        f.write(f"Box-plot quartiles: P{_q1_percentile} / P{_q3_percentile} "
                "(standard IQR, not the old 5/95 misleading non-standard "
                "convention).\n")
        f.write("\nSample names (with group):\n")
        for item in summary_data_list:
            f.write(f"  - {item['sample_name']}  [{item.get('group', '?')}]\n")
        if flagged_high_clamp:
            f.write("\nFlagged (high PSF clamp — kept in aggregate, mean biased upward):\n")
            for row in flagged_high_clamp:
                f.write(f"  - {row['sample_name']}: clamp = "
                        f"{row['clamp_percentage']:.2f}%\n")
        f.write("\nGenerated files:\n")
        if summary_data_list:
            f.write("  - summary_statistics.npz (per-sample quartiles, fences, mean, std)\n")
        if summary_clean:
            f.write("  - sample_weighted_aggregate.yaml (equal weight per sample)\n")
            f.write("  - group_significance.yaml (Kruskal-Wallis + pairwise "
                    "Mann-Whitney U with Bonferroni correction + Cliff's delta)\n")
        if all_thickness_data_before_outliers:
            f.write("  - aggregated_thickness_data.npz (voxel-weighted aggregate, no outlier removal)\n")
        if all_thickness_data_for_violin:
            f.write("\nGroup-comparison figures — publication/ (PNG+PDF+SVG), "
                    "statistics use the MOUSE as the replicate (n = animals):\n")
            f.write("  - group_violin_comparison  ** main figure ** the three "
                    "groups side by side: voxel-distribution violin + inner "
                    "quartile box + per-mouse mean dots + group-mean bar + "
                    "mean-difference/significance brackets\n")
            f.write("  - group_ridgeline   stacked density ridges, one per group "
                    "(distribution shape + shift)\n")
            f.write("  - group_ecdf        cumulative thickness per group "
                    "(read off % thinner than X nm)\n")
            f.write("  - group_estimation  mean difference vs Control + bootstrap "
                    "95%% CI (effect size over p-values for small n)\n")
            f.write("  - mouse_violin      per-MOUSE distribution: one violin per "
                    "animal, grouped/coloured by group (mouse-to-mouse spread)\n")
            f.write("  - mouse_ecdf        per-MOUSE cumulative distribution, one "
                    "line per animal coloured by group\n")
            f.write("  - mouse_summary.yaml          per-animal rollup + per-group means\n")
            f.write("  - group_significance_by_mouse.yaml  tests on MOUSE means "
                    "(** cite this **)\n")
            f.write("  - group_significance_by_image.yaml  tests on IMAGE means "
                    "(transparency; overstates n via pseudoreplication)\n")
            f.write("  - normality_report.yaml       Shapiro-Wilk raw vs log "
                    "(validates the lognormal / censored-MLE assumption)\n")

        if clamp_stats_list:
            f.write("\nPSF clamp activation (% of surface voxels where measured² < PSF²):\n")
            f.write(f"(samples above {PSF_CLAMP_WARNING_PCT}% are FLAGGED — kept "
                    "in the aggregate; the mean thickness is biased upward.)\n")
            for row in clamp_stats_list:
                pct = row.get('clamp_percentage', float('nan'))
                cc = row.get('clamp_count', '?')
                sc = row.get('surface_count', '?')
                flag = " [FLAGGED]" if (isinstance(pct, (int, float))
                                         and pct > PSF_CLAMP_WARNING_PCT) else ""
                f.write(f"  - {row['sample_name']}: {pct:.2f}% "
                        f"({cc}/{sc} surface voxels){flag}\n")

    logging.info("Saved metadata to %s", metadata_file)
    logging.info("Statistical analysis completed successfully")
    logging.info("Multi-bin histograms and aggregated thickness data saved in '%s' directory",
                 _stats_dir)


# --- SLURM-array parallel path --------------------------------------------
# The per-sample work (`calculate_stats_one_sample`) runs one-per-array-task;
# each writes a sidecar under <stats_dir>/_partial/. The reduce step
# (`calculate_stats_reduce`) reads the sidecars and produces the cohort
# outputs. This turns the ~N×(load+mask+plot) serial cost into ~1× wall
# clock (all samples concurrent), the same pattern the morph stage uses.

_PARTIAL_SUBDIR = "_partial"


def calculate_stats_one_sample(_inference_result_path: Path, _stats_dir: Path,
                               _sample_name: str, _clipping: bool,
                               _max_thickness: int = HARD_THICKNESS_LIMIT_NM):
    """Process ONE sample (array-task entry point) for BOTH mask variants.
    The heavy files are loaded once; each variant writes its per-sample plots
    into <stats_dir>/<variant>/<sample>/ and a sidecar
    <stats_dir>/<variant>/_partial/<sample>.npz for the reduce step."""
    _stats_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = _inference_result_path / _sample_name
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    logging.info("Per-sample stats (both mask variants): %s", _sample_name)
    loaded = _load_sample_for_stats(sample_dir, _inference_result_path,
                                    _clipping, _max_thickness)
    safe = _sample_name.replace("/", "_")
    for vname, apply_mask in _MASK_VARIANTS:
        partial_dir = _stats_dir / vname / _PARTIAL_SUBDIR
        partial_dir.mkdir(parents=True, exist_ok=True)
        sidecar = partial_dir / f"{safe}.npz"
        result = (_process_loaded_variant(_sample_name, _stats_dir / vname,
                                          loaded, apply_mask, _max_thickness)
                  if loaded is not None else None)
        if result is None:
            np.savez_compressed(sidecar, meta=json.dumps({"name": _sample_name,
                                                          "empty": True}))
            logging.warning("Sample %s (%s) produced no usable data",
                            _sample_name, vname)
            continue
        meta = json.dumps({"name": result["name"], "group": result["group"],
                           "summary": result["summary"], "clamp": result["clamp"]})
        np.savez_compressed(sidecar, meta=meta,
                            data_clean=result["data_clean"],
                            data_preiqr=result["data_preiqr"])
        logging.info("Wrote sidecar %s", sidecar)


def calculate_stats_reduce(_inference_result_path: Path, _stats_dir: Path,
                           _max_thickness: int = HARD_THICKNESS_LIMIT_NM):
    """Reduce step (single job after the array): for EACH mask variant read
    the per-sample sidecars under <stats_dir>/<variant>/_partial/ and produce
    that variant's cohort outputs + figures + metadata."""
    total_samples = sum(1 for d in _inference_result_path.iterdir()
                        if d.is_dir())
    any_found = False
    for vname, apply_mask in _MASK_VARIANTS:
        partial_dir = _stats_dir / vname / _PARTIAL_SUBDIR
        if not partial_dir.is_dir():
            logging.warning("No _partial/ sidecars in %s — skipping variant",
                            _stats_dir / vname)
            continue
        any_found = True
        results = []
        for sidecar in sorted(partial_dir.glob("*.npz")):
            with np.load(sidecar, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if meta.get("empty"):
                    continue
                results.append({
                    "name": meta["name"], "group": meta["group"],
                    "summary": meta["summary"], "clamp": meta["clamp"],
                    "data_clean": z["data_clean"], "data_preiqr": z["data_preiqr"],
                })
        logging.info("Reduce [%s]: %d sidecars → aggregating", vname, len(results))
        _reduce_and_write(_stats_dir / vname, _inference_result_path,
                          results, total_samples, _mask_applied=apply_mask,
                          _max_thickness=_max_thickness)
    if not any_found:
        raise FileNotFoundError(
            f"No _partial/ sidecars under {_stats_dir}/(with_mask|without_mask). "
            "Run the per-sample stage first.")
