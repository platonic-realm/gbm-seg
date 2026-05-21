"""Expert-comparison metric correctness + pair structure + aggregation.

The full ``calculate_expert_comparison`` driver does file I/O (reads
the labeled-inference output + each annotator's TIFF), which we cover
indirectly by exercising the pure-function building blocks below — they
are what the metric values in the report are computed from.
"""
import math

import numpy as np
import pytest

from src.infer.expert_comparison import (
    _binarise,
    _compare,
    _confusion,
    _crop_stem,
    _find_crop_match,
    _metrics_from_confusion,
    _subsample_pred_to_label_z,
    aggregate,
    compare_crop,
)


def test_binarise_thresholds_at_zero():
    arr = np.array([0, 0.5, 1, -1, 0.0001])
    out = _binarise(arr)
    assert out.dtype == bool
    assert out.tolist() == [False, True, True, False, True]


def test_confusion_counts_match_manual():
    pred = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    label = np.array([[1, 0, 0], [1, 1, 0]], dtype=bool)
    c = _confusion(pred, label)
    # TP: positions where both True  → (0,0), (1,1)         → 2
    # FP: pred True, label False     → (0,1)                 → 1
    # FN: pred False, label True     → (1,0)                 → 1
    # TN: both False                  → (0,2), (1,2)          → 2
    assert c == {'TP': 2, 'FP': 1, 'FN': 1, 'TN': 2}


def test_metrics_from_confusion_perfect_overlap():
    c = {'TP': 10, 'FP': 0, 'FN': 0, 'TN': 100}
    m = _metrics_from_confusion(c)
    assert m['Dice'] == pytest.approx(1.0)
    assert m['IoU'] == pytest.approx(1.0)
    assert m['TPR'] == pytest.approx(1.0)
    assert m['PPV'] == pytest.approx(1.0)
    assert m['specificity'] == pytest.approx(1.0)


def test_metrics_from_confusion_zero_overlap():
    c = {'TP': 0, 'FP': 5, 'FN': 5, 'TN': 100}
    m = _metrics_from_confusion(c)
    assert m['Dice'] == 0.0
    assert m['IoU'] == 0.0
    assert m['TPR'] == 0.0
    assert m['PPV'] == 0.0


def test_metrics_from_confusion_handles_empty_class():
    """Dice/IoU/TPR/PPV degenerate to 0 when the denominator is 0
    (empty crop or empty class), matching the trainer's convention."""
    c = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 100}
    m = _metrics_from_confusion(c)
    assert m['Dice'] == 0.0
    assert m['IoU'] == 0.0
    assert m['TPR'] == 0.0
    assert m['PPV'] == 0.0
    assert m['specificity'] == pytest.approx(1.0)


def test_compare_returns_metrics_and_confusion():
    a = np.array([[1, 1, 0]], dtype=bool)
    b = np.array([[1, 0, 0]], dtype=bool)
    out = _compare(a, b)
    assert set(out) == {'Dice', 'IoU', 'TPR', 'PPV', 'specificity',
                        'TP', 'FP', 'FN', 'TN'}


def test_compare_crop_pair_structure():
    """For N annotators, compare_crop yields N (model-vs-expert) pairs
    plus C(N, 2) inter-rater pairs, all keyed by sorted names."""
    pred = np.ones((2, 2), dtype=bool)
    labels = {
        'David': np.ones((2, 2), dtype=bool),
        'Chris': np.zeros((2, 2), dtype=bool),
        'Robin': np.ones((2, 2), dtype=bool),
    }
    pairs = compare_crop(pred, labels)
    # 3 model-vs-expert + 3 inter-rater = 6 pairs.
    assert set(pairs) == {
        'model_vs_Chris', 'model_vs_David', 'model_vs_Robin',
        'Chris_vs_David', 'Chris_vs_Robin', 'David_vs_Robin',
    }
    # All-foreground vs all-foreground → perfect overlap.
    assert pairs['model_vs_David']['Dice'] == pytest.approx(1.0)
    # All-foreground vs all-background → zero overlap.
    assert pairs['model_vs_Chris']['Dice'] == 0.0
    # Inter-rater: David and Robin both all-foreground → Dice 1.
    assert pairs['David_vs_Robin']['Dice'] == pytest.approx(1.0)


def test_aggregate_handles_single_crop():
    per_crop = {
        'crop_a': {
            'model_vs_Chris': {
                'Dice': 0.8, 'IoU': 0.6, 'TPR': 0.9,
                'PPV': 0.7, 'specificity': 0.95,
                'TP': 80, 'FP': 30, 'FN': 10, 'TN': 1000,
            },
        }
    }
    agg = aggregate(per_crop)
    assert set(agg) == {'model_vs_Chris'}
    # With one sample, mean == min == max and std == 0.
    for metric in ('Dice', 'IoU', 'TPR', 'PPV', 'specificity'):
        v = agg['model_vs_Chris'][metric]
        assert v['mean'] == v['min'] == v['max']
        assert math.isclose(v['std'], 0.0)


def test_aggregate_means_match_numpy_means():
    """Aggregate over multiple crops; mean/std/min/max line up with
    the corresponding numpy reductions."""
    per_crop = {
        f'crop_{i}': {
            'model_vs_Chris': {
                'Dice': float(i) / 10,
                'IoU': float(i) / 20,
                'TPR': 1.0, 'PPV': 1.0, 'specificity': 1.0,
                'TP': i, 'FP': 0, 'FN': 0, 'TN': 100,
            },
        }
        for i in range(5)
    }
    agg = aggregate(per_crop)
    dices = [i / 10 for i in range(5)]
    assert agg['model_vs_Chris']['Dice']['mean'] == pytest.approx(np.mean(dices))
    assert agg['model_vs_Chris']['Dice']['std'] == pytest.approx(np.std(dices))
    assert agg['model_vs_Chris']['Dice']['min'] == pytest.approx(min(dices))
    assert agg['model_vs_Chris']['Dice']['max'] == pytest.approx(max(dices))


def test_aggregate_empty():
    """No crops in, empty dict out — driver should warn and return early
    before this point, but the helper must be robust."""
    assert aggregate({}) == {}


# ---------------------------------------------------------------------------
# _crop_stem: strip annotator-specific tail from a crop filename.
# These names come straight from ds_test_labeled in production.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,stem", [
    # Chris dir
    ("NCWM.AUY380.Series004.Test.Chris_corrected_order.tiff",
     "NCWM.AUY380.Series004"),
    ("NCWM.BDP669.Series008.Test.Chris.tiff",
     "NCWM.BDP669.Series008"),
    ("NCWM.CKM105.Series002.Test.Chris.tiff",
     "NCWM.CKM105.Series002"),
    # David dir — note the missing ".Test." on the third one.
    ("NCWM.AUY380.Series004.Test.ForAnnotation_DUJ.tiff",
     "NCWM.AUY380.Series004"),
    ("NCWM.BDP669.Series008.Test.ForAnnotation_DUJ.tiff",
     "NCWM.BDP669.Series008"),
    ("NCWM.CKM105.Series002.ForAnnotation_DUJ.tiff",
     "NCWM.CKM105.Series002"),
    # Robin dir
    ("NCWM.AUY380.Series004.Test.Robin_corrected_order.tiff",
     "NCWM.AUY380.Series004"),
    ("NCWM.BDP669.Series008.Test.Robin.tiff",
     "NCWM.BDP669.Series008"),
    ("NCWM.CKM105.Series002.Test.Robin.tiff",
     "NCWM.CKM105.Series002"),
])
def test_crop_stem_strips_annotator_tails(filename, stem):
    assert _crop_stem(filename) == stem


def test_crop_stem_no_annotator_token_returns_basename():
    """Filename with no annotator token: return just the basename
    minus the extension (caller can still try exact/glob match)."""
    assert _crop_stem("some_other_volume.tiff") == "some_other_volume"


# ---------------------------------------------------------------------------
# _find_crop_match: cross-annotator filename lookup.
# ---------------------------------------------------------------------------

def test_find_crop_match_exact(tmp_path):
    """An exact filename match short-circuits everything else."""
    (tmp_path / "vol.tiff").write_bytes(b"")
    result = _find_crop_match(tmp_path, "vol.tiff")
    assert result == tmp_path / "vol.tiff"


def test_find_crop_match_prefix_glob(tmp_path):
    """The results-infer-labeled subdir-name pattern drops the .tiff,
    so the annotator file is found via a prefix glob on the bare stem."""
    (tmp_path / "vol.tiff").write_bytes(b"")
    result = _find_crop_match(tmp_path, "vol")
    assert result == tmp_path / "vol.tiff"


def test_find_crop_match_cross_annotator_stem(tmp_path):
    """Chris-derived crop name finds David's renamed file via stem."""
    (tmp_path / "NCWM.AUY380.Series004.Test.ForAnnotation_DUJ.tiff"
     ).write_bytes(b"")
    chris_name = "NCWM.AUY380.Series004.Test.Chris_corrected_order.tiff"
    result = _find_crop_match(tmp_path, chris_name)
    assert result == (
        tmp_path / "NCWM.AUY380.Series004.Test.ForAnnotation_DUJ.tiff")


def test_find_crop_match_returns_none_when_missing(tmp_path):
    assert _find_crop_match(tmp_path, "no-such-thing.tiff") is None


def test_find_crop_match_stem_glob_ignores_non_tiff(tmp_path):
    """A side file (.txt) sharing the stem must NOT be returned —
    the matcher restricts to TIFFs on the stem-glob path."""
    (tmp_path / "NCWM.AUY380.Series004.notes.txt").write_bytes(b"")
    chris_name = "NCWM.AUY380.Series004.Test.Chris.tiff"
    assert _find_crop_match(tmp_path, chris_name) is None


# ---------------------------------------------------------------------------
# _subsample_pred_to_label_z: mirror validation's Z%z_scale==0 masking.
# ---------------------------------------------------------------------------

def test_subsample_pred_passthrough_when_z_matches():
    pred = np.zeros((6, 4, 4), dtype=bool)
    out = _subsample_pred_to_label_z(pred, label_z=6, crop_name="x")
    # Same Z: returned tensor is exactly the input (no copy needed).
    assert out is pred


def test_subsample_pred_integer_ratio_picks_every_z_scaleth_slice():
    """For pred Z=36, label Z=6, z_scale=6 → slices [0, 6, 12, 18, 24, 30]."""
    pred = np.zeros((36, 4, 4), dtype=bool)
    # Mark Z=0,6,12,18,24,30 as True so subsample == all-True (Z=6).
    for z in (0, 6, 12, 18, 24, 30):
        pred[z, :, :] = True
    out = _subsample_pred_to_label_z(pred, label_z=6, crop_name="x")
    assert out.shape == (6, 4, 4)
    assert out.all()


def test_subsample_pred_returns_none_for_non_integer_ratio():
    """A Z that isn't a multiple of label_z indicates a config error
    (e.g. wrong scale_factor); the function should warn and bail out
    rather than silently rescale."""
    pred = np.zeros((10, 4, 4), dtype=bool)
    out = _subsample_pred_to_label_z(pred, label_z=3, crop_name="x")
    assert out is None


def test_subsample_pred_returns_none_for_zero_label_z():
    """An empty label is meaningless; surface it instead of dividing by zero."""
    pred = np.zeros((6, 4, 4), dtype=bool)
    assert _subsample_pred_to_label_z(pred, label_z=0, crop_name="x") is None
