"""5-fold orchestrator: aggregator math + trainer best-metric tracking."""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _aggregate_cv: mean/std/min/max/n across folds for each numeric metric
# ---------------------------------------------------------------------------

def test_aggregate_cv_basic_mean_std():
    from train import _aggregate_cv

    per_fold = [
        {'fold': 0, 'Dice': 0.80, 'Loss': 0.10, 'best_epoch': 2, 'best_step': 100},
        {'fold': 1, 'Dice': 0.84, 'Loss': 0.09, 'best_epoch': 3, 'best_step': 200},
        {'fold': 2, 'Dice': 0.78, 'Loss': 0.12, 'best_epoch': 1, 'best_step': 150},
    ]
    agg = _aggregate_cv(per_fold)

    assert set(agg) == {'Dice', 'Loss'}
    assert agg['Dice']['n'] == 3
    assert abs(agg['Dice']['mean'] - np.mean([0.80, 0.84, 0.78])) < 1e-9
    # ddof=1 sample std
    assert abs(agg['Dice']['std']
               - np.std([0.80, 0.84, 0.78], ddof=1)) < 1e-9
    assert agg['Dice']['min'] == 0.78
    assert agg['Dice']['max'] == 0.84


def test_aggregate_cv_skips_non_numeric_and_epoch_step():
    from train import _aggregate_cv

    per_fold = [{'fold': 0, 'Dice': 0.5, 'best_epoch': 1, 'best_step': 50}]
    agg = _aggregate_cv(per_fold)
    assert 'best_epoch' not in agg
    assert 'best_step' not in agg
    assert 'fold' not in agg
    assert 'Dice' in agg


def test_aggregate_cv_handles_single_fold():
    from train import _aggregate_cv

    per_fold = [{'fold': 0, 'Dice': 0.75}]
    agg = _aggregate_cv(per_fold)
    assert agg['Dice']['n'] == 1
    assert agg['Dice']['std'] == 0.0   # ddof=1 with n=1 → fall back to 0


def test_aggregate_cv_empty_returns_empty():
    from train import _aggregate_cv
    assert _aggregate_cv([]) == {}


def test_aggregate_cv_survives_missing_first_fold():
    """If fold 0 died (placeholder with no metric keys), the aggregate must
    still be computed from the surviving folds. Regression: metric names were
    discovered from per_fold[0] only, yielding an empty aggregate when fold 0
    was the casualty."""
    from train import _aggregate_cv

    per_fold = [
        {'fold': 0},                       # died — no metrics
        {'fold': 1, 'Dice': 0.70, 'JaccardIndex': 0.55},
        {'fold': 2, 'Dice': 0.72, 'JaccardIndex': 0.57},
    ]
    agg = _aggregate_cv(per_fold)
    assert agg['Dice']['n'] == 2
    assert agg['Dice']['mean'] == pytest.approx(0.71)
    assert agg['JaccardIndex']['n'] == 2


# ---------------------------------------------------------------------------
# Unet3DTrainer best-metric tracking
# ---------------------------------------------------------------------------

class _FakeStepper:
    def __init__(self):
        self.steps = 0
    def getSteps(self):
        return self.steps


def _make_trainer():
    """Minimal trainer constructed via __new__ — skip GPU/dataset wiring."""
    from src.train.trainer import Unet3DTrainer
    t = Unet3DTrainer.__new__(Unet3DTrainer)
    t.best_valid_metrics = None
    t.best_valid_epoch = None
    t.best_valid_step = None
    t.stepper = _FakeStepper()
    return t


def test_best_metrics_initially_none():
    t = _make_trainer()
    assert t.best_metrics() is None


def test_best_metrics_updates_on_higher_dice():
    """Simulate two validation cycles — the better Dice wins."""
    t = _make_trainer()
    # First valid cycle: Dice 0.5 at epoch 1, step 100.
    t.stepper.steps = 100
    metrics_a = {'Loss': 0.4, 'Dice': 0.5, 'JaccardIndex': 0.33}
    # Hand-execute the update logic from trainEpoch's tail.
    current = float(metrics_a['Dice'])
    prior = -float('inf')
    if current > prior:
        t.best_valid_metrics = dict(metrics_a)
        t.best_valid_epoch = 1
        t.best_valid_step = t.stepper.getSteps()

    # Second valid cycle: Dice 0.7 at epoch 2, step 200.
    t.stepper.steps = 200
    metrics_b = {'Loss': 0.2, 'Dice': 0.7, 'JaccardIndex': 0.55}
    current = float(metrics_b['Dice'])
    prior = float(t.best_valid_metrics.get('Dice', -float('inf')))
    if current > prior:
        t.best_valid_metrics = dict(metrics_b)
        t.best_valid_epoch = 2
        t.best_valid_step = t.stepper.getSteps()

    best = t.best_metrics()
    assert best is not None
    assert best['Dice'] == pytest.approx(0.7)
    assert best['Loss'] == pytest.approx(0.2)
    assert best['best_epoch'] == 2
    assert best['best_step'] == 200


def test_read_per_fold_best_handles_missing(tmp_path):
    """_read_per_fold_best returns placeholder dicts for missing folds."""
    # Write fold_0 and fold_2, leave fold_1 and fold_3 missing.
    import yaml as _yaml

    from train import _read_per_fold_best
    (tmp_path / 'results-train' / 'fold_0').mkdir(parents=True)
    (tmp_path / 'results-train' / 'fold_2').mkdir(parents=True)
    (tmp_path / 'results-train' / 'fold_0' / 'best_metrics.yaml').write_text(
        _yaml.safe_dump({'Dice': 0.81, 'Loss': 0.10,
                         'best_epoch': 2, 'best_step': 100}))
    (tmp_path / 'results-train' / 'fold_2' / 'best_metrics.yaml').write_text(
        _yaml.safe_dump({'Dice': 0.77, 'Loss': 0.13,
                         'best_epoch': 1, 'best_step': 80}))

    per_fold = _read_per_fold_best(str(tmp_path), _k=4)
    assert len(per_fold) == 4
    assert per_fold[0]['Dice'] == 0.81
    assert per_fold[1] == {'fold': 1}  # placeholder
    assert per_fold[2]['Dice'] == 0.77
    assert per_fold[3] == {'fold': 3}  # placeholder


def test_consolidate_cv_results_writes_yaml_and_npz(tmp_path):
    """consolidate_cv_results emits cv_results.yaml + cv_results.npz."""
    import yaml as _yaml

    from train import consolidate_cv_results

    cfg = {
        'root_path': str(tmp_path),
        'trainer': {'logging': {'wandb': {'enabled': False}}},
    }
    per_fold = [
        {'fold': 0, 'Dice': 0.80, 'Loss': 0.10},
        {'fold': 1, 'Dice': 0.84, 'Loss': 0.09},
        {'fold': 2, 'Dice': 0.78, 'Loss': 0.12},
    ]

    out = consolidate_cv_results(per_fold, cfg, _exp_name='mini')

    yaml_path = tmp_path / 'cv_results.yaml'
    npz_path = tmp_path / 'cv_results.npz'
    assert yaml_path.exists() and npz_path.exists()

    loaded = _yaml.safe_load(yaml_path.read_text())
    assert loaded['k'] == 3
    assert loaded['experiment'] == 'mini'
    assert len(loaded['per_fold']) == 3
    assert 'Dice' in loaded['aggregate']

    import numpy as _np
    npz = _np.load(npz_path)
    assert 'Dice' in npz.files
    assert npz['Dice'].shape == (3,)
    assert out is not None


def test_best_metrics_keeps_better_when_dice_drops():
    """A subsequent worse Dice must NOT overwrite the recorded best."""
    t = _make_trainer()
    t.best_valid_metrics = {'Loss': 0.2, 'Dice': 0.85}
    t.best_valid_epoch = 1
    t.best_valid_step = 100

    # Worse run.
    t.stepper.steps = 200
    worse = {'Loss': 0.5, 'Dice': 0.6}
    current = float(worse['Dice'])
    prior = float(t.best_valid_metrics['Dice'])
    if current > prior:
        t.best_valid_metrics = dict(worse)
        t.best_valid_epoch = 2
        t.best_valid_step = t.stepper.getSteps()

    best = t.best_metrics()
    assert best['Dice'] == pytest.approx(0.85)
    assert best['best_epoch'] == 1
