"""Ablation spec parsing + cell expansion + override merging. Locks in §A2-orch."""

import pytest
import yaml

from src.ablation.spec import Cell, apply_overrides, parse_spec


def _write_spec(tmp_path, doc):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="UTF-8")
    return p


def _good_spec():
    return {
        'study': 'loss_pilot',
        'base_experiment': 'my_baseline',
        'cells': [
            {'name': 'dice', 'overrides': {'trainer.loss': 'Dice'}},
            {'name': 'cont',
             'overrides': {'trainer.loss': 'Cont',
                           'trainer.cont_alpha': 0.7}},
        ],
        'folds': [0, 1],
    }


# --- parse_spec ------------------------------------------------------------

def test_parse_spec_happy_path(tmp_path):
    spec = parse_spec(_write_spec(tmp_path, _good_spec()))
    assert spec.study == 'loss_pilot'
    assert spec.base_experiment == 'my_baseline'
    assert len(spec.cells) == 2
    assert spec.cells[0].name == 'dice'
    assert spec.cells[1].overrides == {'trainer.loss': 'Cont',
                                       'trainer.cont_alpha': 0.7}
    assert spec.folds == [0, 1]


def test_parse_spec_missing_key_raises(tmp_path):
    bad = _good_spec()
    del bad['cells']
    with pytest.raises(ValueError, match="missing required keys"):
        parse_spec(_write_spec(tmp_path, bad))


def test_parse_spec_empty_cells_raises(tmp_path):
    bad = _good_spec()
    bad['cells'] = []
    with pytest.raises(ValueError, match="non-empty list"):
        parse_spec(_write_spec(tmp_path, bad))


def test_parse_spec_duplicate_cell_name_raises(tmp_path):
    bad = _good_spec()
    bad['cells'].append({'name': 'dice', 'overrides': {}})
    with pytest.raises(ValueError, match="duplicate cell name"):
        parse_spec(_write_spec(tmp_path, bad))


def test_parse_spec_invalid_name_raises(tmp_path):
    bad = _good_spec()
    bad['study'] = 'bad/name'
    with pytest.raises(ValueError, match="study"):
        parse_spec(_write_spec(tmp_path, bad))


def test_parse_spec_invalid_fold_raises(tmp_path):
    bad = _good_spec()
    bad['folds'] = [0, -1]
    with pytest.raises(ValueError, match="non-negative"):
        parse_spec(_write_spec(tmp_path, bad))


def test_parse_spec_missing_overrides_is_treated_as_empty(tmp_path):
    """An override-free cell is valid (pins are encoded via missing keys)."""
    doc = _good_spec()
    doc['cells'] = [{'name': 'baseline'}]  # no overrides key
    spec = parse_spec(_write_spec(tmp_path, doc))
    assert spec.cells == [Cell(name='baseline', overrides={})]


# --- Spec.expand -----------------------------------------------------------

def test_expand_produces_cross_product_of_cells_and_folds(tmp_path):
    spec = parse_spec(_write_spec(tmp_path, _good_spec()))
    runs = spec.expand()
    # 2 cells × 2 folds = 4
    assert len(runs) == 4
    names = [r.experiment_name for r in runs]
    assert names == [
        'loss_pilot__dice__fold0',
        'loss_pilot__dice__fold1',
        'loss_pilot__cont__fold0',
        'loss_pilot__cont__fold1',
    ]
    # Each run carries the matching cell + fold.
    assert runs[0].cell.name == 'dice' and runs[0].fold == 0
    assert runs[2].cell.name == 'cont' and runs[2].fold == 0


# --- apply_overrides -------------------------------------------------------

def test_apply_overrides_dotted_path_writes_nested():
    base = {'trainer': {'loss': 'Cont', 'optim': {'name': 'adam', 'lr': 1e-4}}}
    out = apply_overrides(base, {'trainer.loss': 'Dice',
                                  'trainer.optim.lr': 5e-4})
    assert out['trainer']['loss'] == 'Dice'
    assert out['trainer']['optim']['lr'] == 5e-4
    assert out['trainer']['optim']['name'] == 'adam'   # unchanged
    # Original isn't mutated.
    assert base['trainer']['loss'] == 'Cont'
    assert base['trainer']['optim']['lr'] == 1e-4


def test_apply_overrides_creates_missing_intermediate_dicts():
    base = {'trainer': {}}
    out = apply_overrides(base, {'trainer.wandb.enabled': True})
    assert out['trainer']['wandb']['enabled'] is True


def test_apply_overrides_replaces_existing_leaves():
    base = {'a': {'b': 1, 'c': 2}}
    out = apply_overrides(base, {'a.b': 99})
    assert out['a'] == {'b': 99, 'c': 2}


def test_apply_overrides_empty_key_raises():
    with pytest.raises(ValueError, match="non-empty"):
        apply_overrides({}, {'': 1})
