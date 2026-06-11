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
import re
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import tifffile
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

METRIC_NAMES = ('Dice', 'IoU', 'TPR', 'PPV', 'specificity')

# Annotator-specific tokens that separate the volume-identifying "stem"
# (e.g. `NCWM.AUY380.Series004`) from the per-annotator name fragment in a
# crop's filename (e.g. `.Test.Chris_corrected_order.tiff`,
# `.ForAnnotation_DUJ.tiff`, `.Test.Robin.tiff`). The match is on dotted
# segments so a bare `Test` inside another word doesn't trigger.
_ANNOTATOR_TOKEN_RE = re.compile(
    # `\b` is wrong here because `_` is a word character: it wouldn't
    # break between `ForAnnotation` and `_DUJ`. An explicit lookahead
    # for `.`, `_`, or end-of-string covers every observed tail
    # (`.Test.Chris.tiff`, `.ForAnnotation_DUJ.tiff`, `.Test.Robin_corrected_order.tiff`).
    r'\.(?:Test|For_?Annotation|Annotated|Chris|David|Robin)(?=[._]|$)',
    re.IGNORECASE,
)


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
    # 4-channel labelled TIFF: axes (Z, C, H, W); channels 0..2 are
    # nephrin / collagen-4 / WGA intensity and channel 3 is the binary
    # annotator label. See src/data/ds_base.py for the input convention.
    return _binarise(arr[:, 3, :, :])


def _subsample_pred_to_label_z(model_pred: np.ndarray,
                               label_z: int,
                               crop_name: str) -> np.ndarray | None:
    """Subsample a Z-upsampled prediction down to the label's native Z.

    Inference z-upsamples its input (``inference.inference_ds.scale_factor``
    via trilinear interpolation in ds_base.py:64) so the model sees data
    at training-time Z spacing. The expert-annotated crops in
    ``ds_test_labeled`` stay at the native microscope Z — so a Z=6 label
    will face a Z=label_z*scale prediction.

    Mirror the training-validation rule (factory.py:372-379, "Validation
    metrics are masked to slices where Z % z_scale == 0 — i.e. the
    original-label positions"): keep only ``prediction[::z_scale]``,
    starting at index 0. The same label voxel was otherwise represented
    at ``z_scale`` adjacent upsampled positions; this picks one
    representative per original slice.

    Returns the subsampled prediction, or ``None`` (with a logged
    warning) if pred-Z isn't an exact integer multiple of label-Z —
    that points at a Z-axis configuration error worth surfacing rather
    than silently rescaling.
    """
    pred_z = model_pred.shape[0]
    if pred_z == label_z:
        return model_pred
    if label_z == 0 or pred_z % label_z != 0:
        logging.warning(
            "Cannot align prediction Z=%d to label Z=%d for crop %s "
            "(non-integer ratio); skipping this crop.",
            pred_z, label_z, crop_name)
        return None
    z_scale = pred_z // label_z
    return model_pred[::z_scale, :, :]


def _crop_stem(crop_name: str) -> str:
    """Extract the volume-identifying stem from a crop filename, dropping
    the annotator-specific tail.

    The expert dirs use different filename suffixes for the same crop —
    e.g. all three of these describe the same volume:
      Chris/  NCWM.AUY380.Series004.Test.Chris_corrected_order.tiff
      David/  NCWM.AUY380.Series004.Test.ForAnnotation_DUJ.tiff
      Robin/  NCWM.AUY380.Series004.Test.Robin_corrected_order.tiff
    The shared identity is the leading `NCWM.AUY380.Series004` portion.
    This strips everything from the first annotator-token onward, plus
    the file extension.

    Falls back to ``crop_name`` minus the extension if no token matches —
    callers can still exact/glob-match on that.
    """
    no_ext = re.sub(r'\.tiff?$', '', crop_name, flags=re.IGNORECASE)
    m = _ANNOTATOR_TOKEN_RE.search(no_ext)
    return no_ext[:m.start()] if m else no_ext


def _find_crop_match(annotator_dir: Path, crop_name: str) -> Path | None:
    """Find the annotator's TIFF matching ``crop_name``.

    Tries in order: (1) an exact filename match, (2) a glob with
    ``crop_name`` as prefix (handles the case where results-infer-labeled
    dropped a `.tiff` extension), (3) a stem-based glob that ignores the
    per-annotator filename tail. The stem match is what lets a
    Chris-derived crop name find David's and Robin's versions of the
    same volume — see :func:`_crop_stem` for the token list.
    """
    exact = annotator_dir / crop_name
    if exact.exists():
        return exact
    # Tier 2: prefix glob (cheap, common case).
    candidates = sorted(annotator_dir.glob(crop_name + '*'))
    if candidates:
        return candidates[0]
    # Tier 3: stem glob — strip the annotator-specific tail and match the
    # volume identity. Required for cross-annotator lookups.
    stem = _crop_stem(crop_name)
    if stem and stem != crop_name:
        candidates = sorted(annotator_dir.glob(stem + '*'))
        # Keep only real TIFFs to avoid stray matches.
        candidates = [c for c in candidates
                      if c.suffix.lower() in ('.tif', '.tiff')]
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
            if label.shape[1:] != model_pred.shape[1:]:
                logging.warning(
                    "XY mismatch for crop %s vs annotator %s: "
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

        # Mirror the training-validation rule for Z-upsampled inference
        # (factory.py: "validation metrics are masked to slices where
        # Z % z_scale == 0"): keep only the prediction slices at the
        # original-label positions. All annotators of the same crop
        # share Z, so subsample once. ``aligned_pred`` is what gets fed
        # to the metric calls instead of ``model_pred``.
        label_z = next(iter(expert_labels.values())).shape[0]
        aligned_pred = _subsample_pred_to_label_z(
            model_pred, label_z, crop_name)
        if aligned_pred is None:
            continue
        per_crop[crop_name] = compare_crop(aligned_pred, expert_labels)

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
    # dpi 120 -> 480: 4x the linear pixel resolution (publication-quality raster).
    fig.savefig(out_path, dpi=480, bbox_inches='tight')
    plt.close(fig)
    logging.info('Wrote %s', out_path)
