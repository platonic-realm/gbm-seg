# Z-continuity analysis for inference predictions.
#
# Quantifies the along-Z smoothness ("jaggedness") of a predicted GBM mask —
# the quantity the ContLoss continuity term optimises but the validation Dice
# (masked to real-label slices) structurally CANNOT see. Lets a Cont-trained
# model be compared against a CrossEntropy-trained one on the axis the loss
# actually targets.
#
# Modelled on the stats step: a dedicated `gbm.py continuity -it <tag>` pass
# over a directory of inference predictions, dataset-agnostic (validation or
# test). For every sample it scores BOTH the raw model output
# (prediction.npz) and the post-processed reconstruction
# (prediction_psp.npz), then aggregates across samples.

from __future__ import annotations

# Python
import logging
from pathlib import Path

# Library
import numpy as np
import yaml

# The two prediction artifacts scored per sample: raw model output (fair
# loss comparison, no post-processing confound) and the PSP reconstruction
# (the shipped product). (variant_name, filename).
_PRED_VARIANTS = (("raw", "prediction.npz"),
                  ("psp", "prediction_psp.npz"))


def _as_zhw_bool(_arr: np.ndarray) -> np.ndarray:
    """Coerce a loaded prediction to a boolean (Z, H, W) volume. Predictions
    are saved as (D, H, W) with Z on axis 0; squeeze any singleton leading
    channel/batch axis defensively."""
    arr = np.asarray(_arr)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D (Z,H,W) prediction, got shape {arr.shape}")
    return arr.astype(bool)


def continuity_metrics(_mask: np.ndarray) -> dict:
    """Z-continuity metrics for one binary (Z, H, W) volume.

    All three measure along-Z smoothness; lower jaggedness == smoother. The
    per-foreground-voxel normalisation is the fairness control: raw Z-TV is
    trivially small for a model that simply predicts LESS foreground (fewer
    boundaries), so we divide the boundary count by the foreground volume and
    report that volume alongside.
    """
    mask = _as_zhw_bool(_mask)
    z = mask.shape[0]
    fg = int(mask.sum())

    if z < 2 or fg == 0:
        # Degenerate: a single Z-slice or an empty prediction has no along-Z
        # structure to score. Report NaN metrics but a real fg volume so the
        # aggregate can still weight/skip it sensibly.
        return {
            "z_tv_per_voxel": float("nan"),
            "z_tv_per_fg_voxel": float("nan"),
            "mean_flips_per_fg_column": float("nan"),
            "mean_adjacent_iou": float("nan"),
            "fg_voxels": fg,
            "fg_fraction": float(fg / mask.size),
            "z_slices": int(z),
        }

    m = mask.astype(np.int8)
    # |mask[z+1] - mask[z]| summed over the volume == count of along-Z
    # boundary crossings (a voxel that flips 0<->1 between adjacent slices).
    z_diff = np.abs(m[1:] - m[:-1])                       # (Z-1, H, W)
    n_transitions = int(z_diff.sum())

    # 1. Z-TV per voxel: transitions / total adjacent-voxel pairs. Dominated
    #    by volume size; kept for completeness, NOT the headline.
    z_tv_per_voxel = n_transitions / z_diff.size
    # 2. Z-TV per foreground voxel (HEADLINE): boundary crossings normalised
    #    by membrane volume. Fair across models that predict different amounts.
    z_tv_per_fg_voxel = n_transitions / fg

    # 3. Per-column flip count: for each (H,W) column with any foreground,
    #    how many times the label flips along Z. A clean single membrane
    #    crossing == 2 flips; a jagged column has more.
    flips_per_column = np.abs(np.diff(m, axis=0)).sum(axis=0)   # (H, W)
    fg_columns = mask.any(axis=0)                               # (H, W)
    mean_flips = float(flips_per_column[fg_columns].mean())

    # 4. Adjacent-slice IoU: overlap between consecutive Z-slices, averaged
    #    over slice pairs where either is non-empty. High == smooth.
    inter = (mask[1:] & mask[:-1]).sum(axis=(1, 2)).astype(np.float64)
    union = (mask[1:] | mask[:-1]).sum(axis=(1, 2)).astype(np.float64)
    valid = union > 0
    mean_iou = float((inter[valid] / union[valid]).mean()) if valid.any() \
        else float("nan")

    return {
        "z_tv_per_voxel": float(z_tv_per_voxel),
        "z_tv_per_fg_voxel": float(z_tv_per_fg_voxel),
        "mean_flips_per_fg_column": mean_flips,
        "mean_adjacent_iou": mean_iou,
        "fg_voxels": fg,
        "fg_fraction": float(fg / mask.size),
        "z_slices": int(z),
    }


def _analyze_sample(_sample_dir: Path) -> dict | None:
    """Score every prediction variant present for one sample directory.
    Returns {variant: metrics, ...} or None if no prediction files exist."""
    out = {}
    for vname, fname in _PRED_VARIANTS:
        path = _sample_dir / fname
        if not path.exists():
            continue
        try:
            arr = np.load(path)["arr"]
        except Exception as exc:
            logging.warning("Could not load %s (%s); skipping.", path, exc)
            continue
        out[vname] = continuity_metrics(arr)
    return out or None


def _aggregate(_per_sample: list, _variant: str) -> dict:
    """Mean/std/n across samples for each metric of one prediction variant.
    Non-finite per-sample values (degenerate volumes) are excluded."""
    metric_names = ["z_tv_per_voxel", "z_tv_per_fg_voxel",
                    "mean_flips_per_fg_column", "mean_adjacent_iou",
                    "fg_voxels", "fg_fraction"]
    agg = {}
    for name in metric_names:
        vals = np.array(
            [s[_variant][name] for s in _per_sample
             if _variant in s and np.isfinite(s[_variant].get(name, np.nan))],
            dtype=float)
        if vals.size == 0:
            continue
        agg[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": int(vals.size),
        }
    return agg


def calculate_continuity(_inference_result_path: Path,
                         _output_dir: Path) -> dict:
    """Z-continuity analysis over every sample under an inference tag.

    Scores raw + PSP predictions per sample, aggregates across samples, and
    writes ``continuity_result.yaml`` (per-sample + aggregate) under
    ``_output_dir``. Returns the result dict.
    """
    logging.info("Starting Z-continuity analysis")
    logging.info("Input path:  %s", _inference_result_path)
    logging.info("Output path: %s", _output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(d for d in _inference_result_path.iterdir()
                         if d.is_dir())
    logging.info("Found %d samples", len(sample_dirs))

    per_sample = []
    for i, sd in enumerate(sample_dirs, start=1):
        logging.info("Processing %s (%d/%d)", sd.name, i, len(sample_dirs))
        m = _analyze_sample(sd)
        if m is None:
            logging.warning("No prediction files in %s; skipping.", sd.name)
            continue
        m["sample"] = sd.name
        per_sample.append(m)

    aggregate = {v: _aggregate(per_sample, v) for v, _ in _PRED_VARIANTS}
    out = {
        "n_samples": len(per_sample),
        "per_sample": per_sample,
        "aggregate": aggregate,
    }
    yaml_path = _output_dir / "continuity_result.yaml"
    with open(yaml_path, "w", encoding="UTF-8") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    logging.info("Continuity results written to %s", yaml_path)

    # Console summary — the headline jaggedness numbers per variant.
    for v, _ in _PRED_VARIANTS:
        a = aggregate.get(v) or {}
        if "z_tv_per_fg_voxel" in a:
            logging.info(
                "  [%s] Z-TV/fg=%.4f  flips/col=%.3f  adj-IoU=%.4f  (n=%d)",
                v, a["z_tv_per_fg_voxel"]["mean"],
                a["mean_flips_per_fg_column"]["mean"],
                a["mean_adjacent_iou"]["mean"],
                a["z_tv_per_fg_voxel"]["n"])
    return out


def compare_continuity(_runs: dict) -> dict:
    """Tabulate the continuity aggregate of several runs side by side.

    ``_runs`` maps a label (e.g. 'cont', 'crossentropy') to a
    ``continuity_result.yaml`` path. Returns {variant: {metric: {label: mean}}}
    and logs a table — the cont-vs-CE comparison on the axis Dice can't see.
    """
    loaded = {}
    for label, path in _runs.items():
        p = Path(path)
        if not p.exists():
            logging.warning("Missing continuity result for '%s': %s", label, p)
            continue
        with open(p, encoding="UTF-8") as f:
            loaded[label] = yaml.safe_load(f)

    table = {}
    headline = ["z_tv_per_fg_voxel", "mean_flips_per_fg_column",
                "mean_adjacent_iou", "fg_fraction"]
    for variant, _ in _PRED_VARIANTS:
        table[variant] = {}
        for metric in headline:
            row = {}
            for label, res in loaded.items():
                cell = (res.get("aggregate", {}).get(variant, {})
                        .get(metric))
                if cell is not None:
                    row[label] = cell["mean"]
            if row:
                table[variant][metric] = row
        logging.info("=== continuity [%s] ===", variant)
        for metric, row in table[variant].items():
            cells = "  ".join(f"{lbl}={val:.4f}" for lbl, val in row.items())
            logging.info("  %-26s %s", metric, cells)
    return table
