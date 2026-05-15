"""Snapper save/load roundtrip + full-state resume coverage.

Locks in §1.3 (DP/non-DP symmetric save/load), §1.7 (weights_only handling),
and the post-3.x full-state resume contract: model + optimiser + scheduler
+ stepper + scaler + RNG + best-tracker + wandb run id, all in one .pt.
"""

import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DataParallel

from src.train.snapper import Snapper


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class _DummyStepper:
    """Minimal stepper-like object covering the state_dict contract."""
    def __init__(self, steps=0):
        self.steps = steps

    def state_dict(self):
        return {'steps': self.steps, 'scaler': None}

    def load_state_dict(self, state):
        self.steps = int(state.get('steps', 0))


def _state_dicts_equal(a, b):
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


# --- Basic save/load roundtrip (DP/non-DP symmetric) ---------------------


def test_save_load_no_dp(tmp_snapshot_dir):
    src = _TinyModel()
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(src, _epoch=0, _step=1, _async=False)

    saved_path = os.path.join(tmp_snapshot_dir, "000-0001.pt")
    assert os.path.exists(saved_path)

    dst = _TinyModel()
    with torch.no_grad():
        for p in dst.parameters():
            p.fill_(0.0)

    snapper.load(dst, _device='cpu', _path=saved_path)
    assert _state_dicts_equal(src.state_dict(), dst.state_dict())


def test_save_dp_load_into_non_dp(tmp_snapshot_dir):
    """Save under DP, load into a bare model — the §1.3 regression."""
    src = _TinyModel()
    src_dp = DataParallel(src)
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(src_dp, _epoch=0, _step=1, _async=False)

    dst = _TinyModel()
    snapper.load(dst, _device='cpu',
                 _path=os.path.join(tmp_snapshot_dir, "000-0001.pt"))
    assert _state_dicts_equal(src.state_dict(), dst.state_dict())


def test_save_non_dp_load_into_dp(tmp_snapshot_dir):
    src = _TinyModel()
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(src, _epoch=0, _step=1, _async=False)

    dst = _TinyModel()
    dst_dp = DataParallel(dst)
    snapper.load(dst_dp, _device='cpu',
                 _path=os.path.join(tmp_snapshot_dir, "000-0001.pt"))
    assert _state_dicts_equal(src.state_dict(), dst.state_dict())


def test_load_missing_snapshot_returns_none(tmp_snapshot_dir):
    """No `continue/` subdirectory → load returns None (fresh-run path)."""
    snapper = Snapper(tmp_snapshot_dir)
    result = snapper.load(_TinyModel(), _device='cpu')
    assert result is None


def test_init_creates_snapshot_path_directory(tmp_path):
    """The per-fold path `.../snapshots/fold_3` is created on construction
    so the first save target exists (regression for the pre-fix
    FileNotFoundError on the zipfile open)."""
    target = tmp_path / "results-train" / "snapshots" / "fold_3"
    assert not target.exists()
    assert not target.parent.exists()

    snapper = Snapper(str(target))
    assert target.is_dir()

    snapper.save(_TinyModel(), _epoch=0, _step=1, _async=False)
    assert (target / "000-0001.pt").exists()


# --- Full-state resume contract -----------------------------------------


def test_save_records_full_resume_state(tmp_snapshot_dir):
    """A saved .pt must carry every key needed for a clean resume."""
    model = _TinyModel()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, total_iters=10)
    stepper = _DummyStepper(steps=42)

    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(model, _epoch=3, _step=42,
                 _stepper=stepper, _optimizer=optim, _scheduler=sched,
                 _best_valid_metrics={'Dice': 0.7}, _best_valid_epoch=2,
                 _best_valid_step=20, _async=False)

    snap = torch.load(os.path.join(tmp_snapshot_dir, "003-0042.pt"),
                      map_location='cpu', weights_only=False)

    # Top-level keys
    for key in ('MODEL_STATE', 'OPTIMIZER_STATE', 'SCHEDULER_STATE',
                'STEPPER_STATE', 'EPOCH', 'STEP', 'BEST_VALID_METRICS',
                'BEST_VALID_EPOCH', 'BEST_VALID_STEP', 'WANDB_RUN_ID',
                'RNG_PYTHON', 'RNG_NUMPY', 'RNG_TORCH', 'MODEL_CARD'):
        assert key in snap, f"missing key: {key}"

    assert snap['EPOCH'] == 3
    assert snap['STEP'] == 42
    assert snap['STEPPER_STATE']['steps'] == 42
    assert snap['BEST_VALID_METRICS']['Dice'] == 0.7
    assert snap['BEST_VALID_EPOCH'] == 2
    # MODEL_CARD lives in-band (no sibling .yaml any more).
    assert snap['MODEL_CARD']['epoch'] == 3
    assert snap['MODEL_CARD']['step'] == 42
    assert 'created_utc' in snap['MODEL_CARD']
    assert 'torch_version' in snap['MODEL_CARD']
    # Sibling .yaml should NOT exist — the card moved into the .pt.
    assert not os.path.exists(
        os.path.join(tmp_snapshot_dir, "003-0042.yaml"))


def test_load_restores_optimizer_and_scheduler_state(tmp_snapshot_dir):
    """Optimiser Adam moments + scheduler step counter must round-trip."""
    model = _TinyModel()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, total_iters=10)
    # Run a few steps to populate Adam moments + scheduler counter.
    for _ in range(5):
        optim.zero_grad()
        out = model(torch.randn(3, 4))
        out.sum().backward()
        optim.step()
        sched.step()

    saved_optim_state = optim.state_dict()
    saved_sched_state = sched.state_dict()

    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(model, _epoch=0, _step=5,
                 _optimizer=optim, _scheduler=sched, _async=False)

    # Fresh components — different LR / no moments yet.
    new_model = _TinyModel()
    new_optim = torch.optim.Adam(new_model.parameters(), lr=999.0)
    new_sched = torch.optim.lr_scheduler.PolynomialLR(new_optim, total_iters=10)

    snapper.load(new_model, _device='cpu',
                 _path=os.path.join(tmp_snapshot_dir, "000-0005.pt"),
                 _optimizer=new_optim, _scheduler=new_sched)

    # Optimizer state matches (Adam exp_avg / exp_avg_sq tensors round-trip).
    for k in saved_optim_state['state']:
        assert torch.equal(saved_optim_state['state'][k]['exp_avg'],
                           new_optim.state_dict()['state'][k]['exp_avg'])
    # Scheduler step counter matches.
    assert (saved_sched_state['_step_count']
            == new_sched.state_dict()['_step_count'])


def test_load_restores_stepper_state(tmp_snapshot_dir):
    """Stepper.load_state_dict must restore the steps counter exactly."""
    model = _TinyModel()
    src_stepper = _DummyStepper(steps=13000)
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(model, _epoch=10, _step=13000,
                 _stepper=src_stepper, _async=False)

    dst_stepper = _DummyStepper(steps=0)
    snapper.load(model, _device='cpu',
                 _path=os.path.join(tmp_snapshot_dir, "010-13000.pt"),
                 _stepper=dst_stepper)
    assert dst_stepper.steps == 13000


def test_load_returns_resume_info_dict(tmp_snapshot_dir):
    """Load contract: returns the resume dict containing epoch/step/best/wandb."""
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=7, _step=2000,
                 _best_valid_metrics={'Dice': 0.66},
                 _best_valid_epoch=6, _best_valid_step=1500, _async=False)

    info = snapper.load(_TinyModel(), _device='cpu',
                        _path=os.path.join(tmp_snapshot_dir, "007-2000.pt"))
    assert info == {
        'epoch': 7,
        'step': 2000,
        'best_metrics': {'Dice': 0.66},
        'best_epoch': 6,
        'best_step': 1500,
        'wandb_run_id': None,
    }


def test_load_auto_discovery_picks_latest_from_continue_dir(tmp_snapshot_dir):
    """Auto-discovery: load looks in <path>/continue/ and picks the most-recent
    .pt by filename sort. Files in <path>/ itself are intentionally ignored."""
    snapper = Snapper(tmp_snapshot_dir)
    # Two snapshots in the main directory — should NOT be picked up by load().
    snapper.save(_TinyModel(), _epoch=1, _step=100, _async=False)
    snapper.save(_TinyModel(), _epoch=2, _step=200, _async=False)

    # No continue/ subdir yet — load returns None.
    assert snapper.load(_TinyModel(), _device='cpu') is None

    # Move the later snapshot into continue/.
    cont = os.path.join(tmp_snapshot_dir, 'continue')
    os.makedirs(cont, exist_ok=True)
    os.rename(os.path.join(tmp_snapshot_dir, "002-0200.pt"),
              os.path.join(cont, "002-0200.pt"))

    info = snapper.load(_TinyModel(), _device='cpu')
    assert info is not None
    assert info['epoch'] == 2
    assert info['step'] == 200


def test_load_with_missing_components_is_graceful(tmp_snapshot_dir):
    """Inference-style load: pass model only, all other components None.
    Restoring should not raise — model weights only."""
    src = _TinyModel()
    optim = torch.optim.Adam(src.parameters(), lr=1e-4)
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(src, _epoch=0, _step=1, _optimizer=optim, _async=False)

    dst = _TinyModel()
    info = snapper.load(dst, _device='cpu',
                        _path=os.path.join(tmp_snapshot_dir, "000-0001.pt"),
                        _stepper=None, _optimizer=None, _scheduler=None)
    assert info is not None
    assert _state_dicts_equal(src.state_dict(), dst.state_dict())


def test_load_restores_rng_states(tmp_snapshot_dir):
    """RNG round-trip: after save + reset + load, draws must match the
    deterministic sequence captured at save time."""
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=0, _step=1, _async=False)
    expected_py = [random.random() for _ in range(3)]
    expected_np = np.random.rand(3).tolist()
    expected_torch = torch.rand(3).tolist()

    # Reset RNGs to a different seed.
    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)

    snapper.load(_TinyModel(), _device='cpu',
                 _path=os.path.join(tmp_snapshot_dir, "000-0001.pt"))
    got_py = [random.random() for _ in range(3)]
    got_np = np.random.rand(3).tolist()
    got_torch = torch.rand(3).tolist()

    assert got_py == expected_py
    assert got_np == expected_np
    assert got_torch == expected_torch


def test_load_skip_rng_when_restore_rng_false(tmp_snapshot_dir):
    """Inference path passes _restore_rng=False to avoid clobbering the
    caller's RNG state with the snapshot's training-time state."""
    random.seed(123)
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=0, _step=1, _async=False)

    random.seed(9999)
    expected = random.random()
    random.seed(9999)
    snapper.load(_TinyModel(), _device='cpu',
                 _path=os.path.join(tmp_snapshot_dir, "000-0001.pt"),
                 _restore_rng=False)
    assert random.random() == expected
