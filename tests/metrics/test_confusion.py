"""Confusion-matrix + derived-metrics regression.

Locks in §1.1 (Pervalence operator precedence) and §1.8 (vectorised CM
that replaced the buggy class-0 remap + nested-loop helpers).
"""

import math

import torch

from src.utils.metrics.clfication import Metrics


def _build(C: int, labels, preds):
    labels_t = torch.tensor(labels)
    preds_t = torch.tensor(preds)
    return Metrics(C, preds_t, labels_t)


def test_binary_perfect_overlap():
    m = _build(2, [0, 1, 1, 0], [0, 1, 1, 0])
    assert m.confusion_matrix.tolist() == [[2, 0], [0, 2]]
    assert m.TruePositive(1).item() == 2
    assert m.FalsePositive(1).item() == 0
    assert m.FalseNegative(1).item() == 0
    assert m.TrueNegative(1).item() == 2
    assert m.Accuracy().item() == 1.0
    assert m.Dice(1).item() == 1.0
    assert m.JaccardIndex(1).item() == 1.0


def test_binary_balanced_half_correct():
    # 2x2 confusion with TP=FN=FP=TN=1.
    m = _build(2, [1, 1, 0, 0], [1, 0, 1, 0])
    assert m.TruePositive(1).item() == 1
    assert m.FalseNegative(1).item() == 1
    assert m.FalsePositive(1).item() == 1
    assert m.TrueNegative(1).item() == 1
    assert m.AllPostive(1).item() == 2
    assert m.AllNegative(1).item() == 2
    assert m.Accuracy().item() == 0.5
    assert m.Dice(1).item() == 0.5
    assert m.TruePositiveRate(1).item() == 0.5
    assert m.PositivePredictiveValue(1).item() == 0.5


def test_pervalence_regression():
    """Pre-fix: ``return AP / AP + AN`` evaluated as ``(AP/AP) + AN = 1 + AN``.

    For AP=2 / AN=2 the buggy code returned 3.0; the correct value is 0.5.
    """
    m = _build(2, [1, 1, 0, 0], [1, 0, 1, 0])
    prevalence = m.Pervalence(1).item()
    assert math.isclose(prevalence, 0.5, abs_tol=1e-6), (
        f"Pervalence operator-precedence bug regressed: got {prevalence}")
    assert prevalence < 1.0  # explicit guard against the old behaviour


def test_class_zero_not_special_cased():
    """The old code remapped predictions==0 to ``number_of_classes``.

    The vectorised replacement should treat all classes symmetrically.
    Build a case with TP for *class 0* and verify it counts correctly.
    """
    # labels=[0,0,1,1], preds=[0,0,1,1] -> TP(0)=2, TP(1)=2
    m = _build(2, [0, 0, 1, 1], [0, 0, 1, 1])
    assert m.TruePositive(0).item() == 2
    assert m.TruePositive(1).item() == 2
    assert m.FalsePositive(0).item() == 0
    assert m.FalsePositive(1).item() == 0


def test_three_class():
    # labels=[0,1,2,0,1,2], preds=[0,1,2,1,2,0]
    # CM[t,p]: [[1,1,0],[0,1,1],[1,0,1]]
    m = _build(3, [0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0])
    assert m.confusion_matrix.tolist() == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    # Class 1: TP=1, FP=row0[1]+row2[1] = 1+0=1, FN=row1[0]+row1[2]=0+1=1
    assert m.TruePositive(1).item() == 1
    assert m.FalsePositive(1).item() == 1
    assert m.FalseNegative(1).item() == 1
    # Total = 6, TP for all classes = 1+1+1 = 3
    assert m.Accuracy().item() == 0.5


def test_4d_input_shape_handled():
    """Real model output is (B, D, H, W); the CM builder must flatten."""
    labels = torch.zeros(2, 4, 8, 8, dtype=torch.long)
    preds = torch.zeros(2, 4, 8, 8, dtype=torch.long)
    labels[..., :4] = 1
    preds[..., :4] = 1
    m = Metrics(2, preds, labels)
    # Half the voxels are class 1 and predicted class 1 -> Dice 1.0
    assert m.Dice(1).item() == 1.0


def test_report_metrics_returns_known_keys():
    m = _build(2, [1, 1, 0, 0], [1, 0, 1, 0])
    out = m.reportMetrics(['Loss', 'Dice', 'JaccardIndex',
                           'TruePositiveRate', 'PositivePredictiveValue'],
                          _loss=torch.tensor(0.42))
    assert set(out.keys()) == {
        'Loss', 'Dice', 'JaccardIndex',
        'TruePositiveRate', 'PositivePredictiveValue'}
    assert math.isclose(out['Dice'].item(), 0.5)
