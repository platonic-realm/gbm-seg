"""All-data (no-holdout) training mode — Stage B.

Exercises the ``validation_loader=None`` path of ``Unet3DTrainer``:
``trainEpoch`` must not touch the absent validation loader, must still
advance an epoch-driven scheduler once per epoch, and ``train()`` must
honour ``starting_epoch``.

The heavy trainStep/Metrics/snapshot path is intentionally NOT exercised
here (it needs a real model + dataset); the empty-training-loader fixture
keeps the test CPU-only and fast. The full all-data loop is
integration-tested by the Stage 0.5 pre-flight run.
"""

import torch
from torch import nn

from src.train.trainer import Unet3DTrainer


def _make_trainer(validation_loader, epochs=3, starting_epoch=0):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=epochs)
    # Empty training loader → trainEpoch's per-batch body never runs, so
    # the trainStep/Metrics/snapper path is skipped; this isolates the
    # validation-None guard and the epoch-end scheduler step.
    train_loader = []
    return Unet3DTrainer(
        model, None, None, None, None, None, scheduler,
        train_loader, validation_loader, 2, ['Dice'], 'cpu',
        _freq=1, _epochs=epochs, _starting_epoch=starting_epoch)


def test_trainer_accepts_no_validation_loader():
    """Constructing with validation_loader=None must not raise."""
    trainer = _make_trainer(validation_loader=None)
    assert trainer.validation_loader is None
    assert trainer.best_valid_metrics is None


def test_train_epoch_none_validation_steps_scheduler_per_epoch():
    """With no validation loader, an epoch-driven scheduler advances once
    per trainEpoch call — it cannot step inside a validation cycle."""
    trainer = _make_trainer(validation_loader=None, epochs=3)
    before = trainer.scheduler.last_epoch
    trainer.trainEpoch(1)            # _epoch > 0 → scheduler steps
    assert trainer.scheduler.last_epoch == before + 1
    # best-valid tracker untouched (no validation ran).
    assert trainer.best_valid_metrics is None


def test_train_epoch_none_validation_epoch0_no_scheduler_step():
    """Epoch 0 does not step the scheduler — matches the fold path's
    `_epoch > 0` guard."""
    trainer = _make_trainer(validation_loader=None, epochs=3)
    before = trainer.scheduler.last_epoch
    trainer.trainEpoch(0)
    assert trainer.scheduler.last_epoch == before


def test_train_epoch_none_scheduler_does_not_crash():
    """scheduler=None (disabled) + no validation must not crash trying to
    step a None scheduler."""
    trainer = _make_trainer(validation_loader=None, epochs=3)
    trainer.scheduler = None
    trainer.trainEpoch(1)            # must not raise


def test_train_respects_starting_epoch():
    """train() iterates range(starting_epoch, epochs + 1) so a resumed
    all-data run skips already-completed epochs."""
    trainer = _make_trainer(validation_loader=None, epochs=4,
                            starting_epoch=2)
    seen = []
    trainer.trainEpoch = lambda e: seen.append(e)
    trainer.train()
    assert seen == [2, 3, 4]


def test_best_metrics_none_without_validation():
    """best_metrics() returns None when no validation cycle ever ran."""
    trainer = _make_trainer(validation_loader=None)
    trainer.trainEpoch(1)
    assert trainer.best_metrics() is None
