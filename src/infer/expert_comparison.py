"""Expert-comparison metrics for the labeled-inference branch.

Compares the model's predictions on ds_test_labeled crops to each
annotator's annotation, and computes inter-rater agreement metrics
across all annotator pairs. Invoked from ``gbm.py stats`` (via
``exper.stats``) when ``results-infer-labeled/<tag>/`` is present on
disk alongside ``results-infer/<tag>/``.

Per crop, for every pair (model vs each expert + each expert vs each
other expert) it computes:
  * Overlap metrics — Dice, IoU, TPR (sensitivity), PPV (precision),
    specificity.
  * Per-class confusion — TP / FP / FN / TN voxel counts.

Aggregate across crops: mean / std / min / max per metric per pair.

Outputs live alongside the morph stats so the two report sections are
independent:
  * expert_comparison.yaml          — per-crop + aggregate
  * expert_comparison_summary.yaml  — aggregate only (smaller)
  * expert_comparison.png           — box plot per metric, one box per pair
"""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import tifffile
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

METRIC_NAMES = ('Dice', 'IoU', 'TPR', 'PPV', 'specificity')


def _binarise(arr) -> np.ndarray:
    """Cast to bool: anything > 0 is foreground."""
    return arr > 0


def _confusion(pred_bool: np.ndarray, label_bool: np.ndarray) -> dict:
    """Per-voxel confusion counts. int64 to never overflow on large volumes."""
    return {
        'TP': int(np.sum(pred_bool & label_bool,   dtype=np.int64)),
        'FP': int(np.sum(pred_bool & ~label_bool,  dtype=np.int64)),
        'FN': int(np.sum(~pred_bool & label_bool,  dtype=np.int64)),
        'TN': int(np.sum(~pred_bool & ~label_bool, dtype=np.int64)),
    }


def _metrics_from_confusion(c: dict) -> dict:
    """Compute Dice/IoU/TPR/PPV/specificity from the {TP, FP, FN, TN} counts.

    Each metric falls back to 0.0 when its denominator is 0 (an empty
    crop or an empty class) — that matches the trainer's convention.
    """
    tp, fp, fn, tn = c['TP'], c['FP'], c['FN'], c['TN']
    dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    tpr = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    ppv = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {'Dice': dice, 'IoU': iou, 'TPR': tpr,
            'PPV': ppv, 'specificity': spec}


def _compare(a_mask: np.ndarray, b_mask: np.ndarray) -> dict:
    """All metrics + the raw confusion counts for one binary mask pair."""
    confusion = _confusion(a_mask, b_mask)
    return {**_metrics_from_confusion(confusion), **confusion}


def compare_crop(model_pred: np.ndarray,
                 expert_labels: dict) -> dict:
    """One crop's worth of pairwise comparisons.

    Pairs computed:
      * model vs each annotator                — N pairs
      * each annotator vs each other annotator — C(N, 2) pairs

    Returns ``{pair_name: metrics_dict}``. Pair names are sorted
    alphabetically within each pair to make output deterministic
    independent of dict-iteration order.
    """
    pairs = {}
    for name in sorted(expert_labels):
        pairs[f'model_vs_{name}'] = _compare(model_pred, expert_labels[name])
    for a, b in combinations(sorted(expert_labels), 2):
        pairs[f'{a}_vs_{b}'] = _compare(expert_labels[a], expert_labels[b])
    return pairs


def aggregate(per_crop: dict) -> dict:
    """Aggregate per-crop metrics across all crops.

    Returns ``{pair_name: {metric: {mean, std, min, max}}}``.
    """
    if not per_crop:
        return {}
    agg = {}
    pair_names = list(next(iter(per_crop.values())).keys())
    for pair in pair_names:
        agg[pair] = {}
        for metric in METRIC_NAMES:
            values = [per_crop[crop][pair][metric] for crop in per_crop]
            agg[pair][metric] = {
                'mean': float(np.mean(values)),
                'std':  float(np.std(values)),
                'min':  float(np.min(values)),
                'max':  float(np.max(values)),
            }
    return agg


def _load_expert_label(tiff_path: Path) -> np.ndarray:
    """Read channel-3 (label) from a 4-channel ZCYX TIFF as bool."""
    with tifffile.TiffFile(str(tiff_path)) as tiff:
        arr = tiff.asarray()
    # 4-channel TIFF: (Z, C, H, W) per CLAUDE.md, channel 3 = label.
    return _binarise(arr[:, 3, :, :])


def _find_crop_match(annotator_dir: Path, crop_name: str) -> Path | None:
    """Find the annotator's TIFF matching `crop_name` (results-infer-labeled
    drops the .tiff extension on its subdir names; the annotator dirs
    keep the original .tiff)."""
    exact = annotator_dir / crop_name
    if exact.exists():
        return exact
    candidates = sorted(annotator_dir.glob(crop_name + '*'))
    if candidates:
        return candidates[0]
    return None


def calculate_expert_comparison(
        _labeled_inference_path: Path,
        _ds_test_labeled_path: Path,
        _stats_dir: Path) -> None:
    """Run the full expert comparison and write YAML + plot outputs.

    Parameters
    ----------
    _labeled_inference_path : Path
        ``results-infer-labeled/<tag>/`` — each crop sub-directory holds
        the model's ``prediction_psp.npz``.
    _ds_test_labeled_path : Path
        ``<exp>/datasets/ds_test_labeled/`` — one sub-directory per
        annotator (e.g. Chris, David, Robin), each holding 4-channel
        ``(Z, C, H, W)`` TIFFs with the annotator's binary label in
        channel 3.
    _stats_dir : Path
        Where to write ``expert_comparison.yaml`` /
        ``expert_comparison_summary.yaml`` / ``expert_comparison.png``.
    """
    labeled_path = Path(_labeled_inference_path)
    annotator_dirs = sorted(
        d for d in Path(_ds_test_labeled_path).iterdir() if d.is_dir())
    if not annotator_dirs:
        logging.warning(
            "No annotator sub-directories under %s; skipping expert "
            "comparison.", _ds_test_labeled_path)
        return
    annotators = [d.name for d in annotator_dirs]
    logging.info("Expert comparison: %d annotators (%s)",
                 len(annotators), ', '.join(annotators))

    crop_dirs = sorted(d for d in labeled_path.iterdir() if d.is_dir())
    if not crop_dirs:
        logging.warning(
            "No crop directories under %s; skipping expert comparison.",
            labeled_path)
        return

    per_crop: dict = {}
    for crop_dir in crop_dirs:
        crop_name = crop_dir.name
        pred_path = crop_dir / 'prediction_psp.npz'
        if not pred_path.exists():
            logging.warning(
                "No prediction_psp.npz in %s; skipping this crop.", crop_dir)
            continue
        with np.load(pred_path) as d:
            model_pred = _binarise(d[list(d.keys())[0]])

        # Load each annotator's label for this crop. Skip the annotator
        # cleanly if their crop is missing or shape-mismatched — partial
        # coverage is still better than aborting the whole report.
        expert_labels: dict = {}
        for annotator_dir in annotator_dirs:
            tiff_path = _find_crop_match(annotator_dir, crop_name)
            if tiff_path is None:
                logging.warning(
                    "No crop matching '%s' under %s — skipping annotator "
                    "for this crop.", crop_name, annotator_dir)
                continue
            label = _load_expert_label(tiff_path)
            if label.shape != model_pred.shape:
                logging.warning(
                    "Shape mismatch for crop %s vs annotator %s: "
                    "prediction %s, label %s — skipping this pair.",
                    crop_name, annotator_dir.name,
                    model_pred.shape, label.shape)
                continue
            expert_labels[annotator_dir.name] = label

        if not expert_labels:
            logging.warning(
                "No usable annotator labels for crop %s; skipping.",
                crop_name)
            continue
        per_crop[crop_name] = compare_crop(model_pred, expert_labels)

    if not per_crop:
        logging.warning("No crops produced comparable pairs; nothing to "
                        "write.")
        return

    agg = aggregate(per_crop)

    _stats_dir.mkdir(parents=True, exist_ok=True)
    full_path = _stats_dir / 'expert_comparison.yaml'
    summary_path = _stats_dir / 'expert_comparison_summary.yaml'
    with open(full_path, 'w') as f:
        yaml.safe_dump({'per_crop': per_crop, 'aggregate': agg}, f,
                       sort_keys=False)
    with open(summary_path, 'w') as f:
        yaml.safe_dump(agg, f, sort_keys=False)
    logging.info("Wrote expert comparison: %s + %s", full_path, summary_path)

    _plot_box(per_crop, _stats_dir / 'expert_comparison.png')


def _plot_box(per_crop: dict, out_path: Path) -> None:
    """One subplot per metric, one box per pair. Pairs along the x-axis,
    metric values on the y-axis. Shared y-axis (all metrics are in [0, 1])
    for easy visual comparison.
    """
    if not per_crop:
        return
    pair_names = list(next(iter(per_crop.values())).keys())
    fig, axes = plt.subplots(1, len(METRIC_NAMES),
                             figsize=(4 * len(METRIC_NAMES), 6),
                             sharey=True)
    for ax, metric in zip(axes, METRIC_NAMES):
        data = [[per_crop[crop][pair][metric] for crop in per_crop]
                for pair in pair_names]
        ax.boxplot(data, tick_labels=pair_names)
        ax.set_title(metric)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('Expert comparison — pairwise agreement across crops',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logging.info('Wrote %s', out_path)
