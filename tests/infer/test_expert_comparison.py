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
    _metrics_from_confusion,
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
