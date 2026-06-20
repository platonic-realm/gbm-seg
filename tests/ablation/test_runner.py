"""Ablation runner materialisation regression. Locks in §A2-orch.

Tests that ``materialise`` correctly:
 - errors when the base experiment is missing
 - creates one cell dir per (cell, fold)
 - symlinks datasets/ and code/ from the base (no large copies)
 - copies fold_assignments.yaml + provenance files
 - writes a cell-specific configs.yaml with overrides applied and a unique
   wandb.run_name
 - is idempotent (existing cells are left alone)

Also tests ``emit_commands`` formatting (plain + sbatch-wrapped).
"""

import os
from pathlib import Path

import pytest
import yaml

from src.ablation.runner import emit_commands, materialise
from src.ablation.spec import Cell, Spec


def _base_configs():
    return {
        'root_path': '/some/experiments/baseline/',
        'trainer': {
            'optimization': {
                'loss': {'name': 'Cont', 'cont_alpha': 0.7},
                'optim': {'name': 'adam', 'lr': 1e-4},
            },
            'logging': {'wandb': {'enabled': False}},
        },
        'inference': {'stitching': 'gaussian'},
    }


def _make_base_experiment(experiments_root: Path):
    """Build a minimal base experiment dir with the files materialise() needs."""
    base = experiments_root / 'baseline'
    base.mkdir(parents=True)
    (base / 'configs.yaml').write_text(
        yaml.safe_dump(_base_configs(), sort_keys=False),
        encoding="UTF-8")
    (base / 'datasets').mkdir()
    (base / 'datasets' / 'ds_train').mkdir()
    (base / 'datasets' / 'ds_train' / 'placeholder.tif').write_text('mock')
    (base / 'code').mkdir()
    (base / 'fold_assignments.yaml').write_text("folds: []\n", encoding="UTF-8")
    (base / 'requirements.txt').write_text("torch==2.6.0\n", encoding="UTF-8")
    (base / 'git_sha.txt').write_text("deadbeef\n", encoding="UTF-8")
    return base


def _make_spec(cells, folds=None):
    return Spec(
        study='loss_pilot',
        base_experiment='baseline',
        cells=cells,
        folds=folds or [0],
    )


# --- materialise -----------------------------------------------------------

def test_materialise_missing_base_raises(tmp_path):
    spec = _make_spec([Cell(name='dice', overrides={'trainer.optimization.loss.name': 'Dice'})])
    with pytest.raises(FileNotFoundError, match="Base experiment not found"):
        materialise(spec, tmp_path)


def test_materialise_creates_one_dir_per_cell_fold(tmp_path):
    _make_base_experiment(tmp_path)
    spec = _make_spec(
        cells=[
            Cell(name='dice', overrides={'trainer.optimization.loss.name': 'Dice'}),
            Cell(name='cont', overrides={'trainer.optimization.loss.name': 'Cont'}),
        ],
        folds=[0, 1],
    )
    paths = materialise(spec, tmp_path)
    assert len(paths) == 4
    expected_names = {
        'loss_pilot__dice__fold0', 'loss_pilot__dice__fold1',
        'loss_pilot__cont__fold0', 'loss_pilot__cont__fold1',
    }
    assert {p.name for p in paths} == expected_names
    for p in paths:
        assert p.is_dir()


def test_materialise_reroots_cell_at_its_own_dir(tmp_path):
    """Each cell's configs.yaml must carry root_path pointing at the CELL dir,
    not the base experiment — otherwise every cell writes results/snapshots
    into the base and sibling cells collide there (regression)."""
    _make_base_experiment(tmp_path)
    spec = _make_spec(
        cells=[
            Cell(name='cont', overrides={'trainer.optimization.loss.name': 'Cont'}),
            Cell(name='crossentropy',
                 overrides={'trainer.optimization.loss.name': 'CrossEntropy'}),
        ],
        folds=[0],
    )
    paths = materialise(spec, tmp_path)
    for p in paths:
        cfg = yaml.safe_load((p / 'configs.yaml').read_text())
        assert cfg['root_path'].rstrip('/') == str(p), (
            f"cell {p.name} root_path={cfg['root_path']} should point at itself")
    # sibling cells must NOT share a root_path
    roots = {yaml.safe_load((p / 'configs.yaml').read_text())['root_path']
             for p in paths}
    assert len(roots) == len(paths)


def test_materialise_symlinks_datasets_and_code(tmp_path):
    base = _make_base_experiment(tmp_path)
    spec = _make_spec([Cell(name='dice', overrides={'trainer.optimization.loss.name': 'Dice'})])
    paths = materialise(spec, tmp_path)
    cell = paths[0]

    # datasets/ and code/ are symlinks, not copies.
    assert (cell / 'datasets').is_symlink()
    assert (cell / 'code').is_symlink()
    # And they resolve to the base experiment's dirs.
    assert os.readlink(cell / 'datasets').endswith(str(base / 'datasets'))


def test_materialise_copies_small_provenance_files(tmp_path):
    _make_base_experiment(tmp_path)
    spec = _make_spec([Cell(name='dice', overrides={'trainer.optimization.loss.name': 'Dice'})])
    paths = materialise(spec, tmp_path)
    cell = paths[0]
    for name in ('fold_assignments.yaml', 'requirements.txt', 'git_sha.txt'):
        assert (cell / name).is_file(), f"{name} should be a copy in the cell"
        assert not (cell / name).is_symlink()


def test_materialise_writes_cell_configs_with_overrides_applied(tmp_path):
    _make_base_experiment(tmp_path)
    spec = _make_spec([
        Cell(name='dice',
             overrides={'trainer.optimization.loss.name': 'Dice',
                        'trainer.optimization.loss.cont_alpha': 1.0}),
    ])
    paths = materialise(spec, tmp_path)
    cfg = yaml.safe_load((paths[0] / 'configs.yaml').read_text(encoding="UTF-8"))
    optimization = cfg['trainer']['optimization']
    assert optimization['loss']['name'] == 'Dice'        # override applied
    assert optimization['loss']['cont_alpha'] == 1.0
    assert optimization['optim']['name'] == 'adam'       # unchanged from base
    # Each cell's W&B run_name is set to the experiment name so cells show up
    # as separate runs in the UI.
    assert (cfg['trainer']['logging']['wandb']['run_name']
            == 'loss_pilot__dice__fold0')
    # Ablation provenance recorded inline for runtime visibility.
    assert cfg['ablation']['cell'] == 'dice'
    assert cfg['ablation']['fold'] == 0


def test_materialise_is_idempotent(tmp_path, caplog):
    import logging
    _make_base_experiment(tmp_path)
    spec = _make_spec([Cell(name='dice', overrides={'trainer.optimization.loss.name': 'Dice'})])
    paths1 = materialise(spec, tmp_path)
    with caplog.at_level(logging.INFO):
        paths2 = materialise(spec, tmp_path)
    assert paths1 == paths2
    assert "already exists" in caplog.text.lower()


# --- emit_commands ---------------------------------------------------------

def test_emit_commands_plain_python(tmp_path):
    _make_base_experiment(tmp_path)
    spec = _make_spec(
        cells=[Cell(name='dice', overrides={'trainer.optimization.loss.name': 'Dice'})],
        folds=[0, 1],
    )
    paths = materialise(spec, tmp_path)
    cmds = emit_commands(spec, paths)
    assert cmds == [
        'python gbm.py train loss_pilot__dice__fold0 --fold 0',
        'python gbm.py train loss_pilot__dice__fold1 --fold 1',
    ]


def test_emit_commands_sbatch_wrapper(tmp_path):
    _make_base_experiment(tmp_path)
    spec = _make_spec(
        cells=[Cell(name='dice', overrides={'trainer.optimization.loss.name': 'Dice'})],
    )
    paths = materialise(spec, tmp_path)
    cmds = emit_commands(spec, paths,
                         sbatch_wrapper='./sbatch/train.sbatch')
    assert cmds == ['sbatch ./sbatch/train.sbatch loss_pilot__dice__fold0 0']
