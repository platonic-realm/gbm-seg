"""Experiment deletion + the optional W&B run cleanup (`gbm.py delete -w`).

wandb is not installed in this venv, so the W&B-cleanup tests inject a
stub `wandb` module into sys.modules and verify `_delete_wandb_runs`
matches runs by their W&B `group` and never raises — a W&B failure must
not block the local experiment deletion.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
import yaml


def _make_experiment(root, name, project='gbm-test', entity='ent'):
    """Create a minimal experiment dir with a configs.yaml."""
    exp = root / name
    exp.mkdir(parents=True)
    cfg = {'trainer': {'wandb': {'project': project, 'entity': entity}}}
    (exp / 'configs.yaml').write_text(yaml.safe_dump(cfg))
    (exp / 'somefile.txt').write_text('data')
    return exp


def _run(group, run_id):
    r = MagicMock()
    r.group = group
    r.id = run_id
    return r


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = types.ModuleType('wandb')
    fake.Api = MagicMock()
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    yield fake


# --- delete_experiment ----------------------------------------------------


def test_delete_experiment_requires_force(tmp_path):
    from src.utils.exper import delete_experiment
    _make_experiment(tmp_path, 'exp1')
    with pytest.raises(RuntimeError, match='force'):
        delete_experiment('exp1', str(tmp_path), _force=False)
    assert (tmp_path / 'exp1').exists()  # refused → still there


def test_delete_experiment_missing_raises(tmp_path):
    from src.utils.exper import delete_experiment
    with pytest.raises(FileNotFoundError):
        delete_experiment('nope', str(tmp_path), _force=True)


def test_delete_experiment_force_removes_dir(tmp_path):
    from src.utils.exper import delete_experiment
    _make_experiment(tmp_path, 'exp1')
    delete_experiment('exp1', str(tmp_path), _force=True)
    assert not (tmp_path / 'exp1').exists()


# --- _delete_wandb_runs ---------------------------------------------------


def test_delete_wandb_runs_deletes_only_matching_group(tmp_path, fake_wandb):
    """Only runs whose W&B group equals the experiment name are deleted."""
    from src.utils.exper import _delete_wandb_runs
    exp = _make_experiment(tmp_path, 'expA')
    mine = [_run('expA', 'a1'), _run('expA', 'a2')]
    other = [_run('expB', 'b1'), _run(None, 'c1')]
    fake_wandb.Api.return_value.runs.return_value = mine + other

    _delete_wandb_runs('expA', str(exp / 'configs.yaml'))

    for r in mine:
        r.delete.assert_called_once()
    for r in other:
        r.delete.assert_not_called()


def test_delete_wandb_runs_missing_configs_is_graceful(tmp_path):
    """No configs.yaml → cannot resolve the project; warn, do not raise."""
    from src.utils.exper import _delete_wandb_runs
    _delete_wandb_runs('ghost', str(tmp_path / 'ghost' / 'configs.yaml'))


def test_delete_wandb_runs_api_failure_is_graceful(tmp_path, fake_wandb):
    """A W&B API failure (e.g. project gone) must not raise — the local
    experiment deletion must still be able to proceed."""
    from src.utils.exper import _delete_wandb_runs
    exp = _make_experiment(tmp_path, 'expA')
    fake_wandb.Api.return_value.runs.side_effect = ValueError('no project')
    _delete_wandb_runs('expA', str(exp / 'configs.yaml'))  # must not raise


def test_delete_experiment_with_wandb_flag(tmp_path, fake_wandb):
    """delete_experiment(_remove_wandb=True) deletes the matching W&B runs
    and then removes the directory."""
    from src.utils.exper import delete_experiment
    exp = _make_experiment(tmp_path, 'expA')
    r = _run('expA', 'a1')
    fake_wandb.Api.return_value.runs.return_value = [r]

    delete_experiment('expA', str(tmp_path), _force=True, _remove_wandb=True)

    r.delete.assert_called_once()
    assert not exp.exists()
