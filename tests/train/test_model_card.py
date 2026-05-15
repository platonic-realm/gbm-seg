"""Per-snapshot model-card regression. Locks in §E4 (provenance card)
plus the post-3.x change: the card now lives **inside the .pt** under the
``MODEL_CARD`` key (sibling ``.yaml`` retired) so a single file contains
everything needed to identify and resume a snapshot.

The model card is a *best-effort* artifact: snapshot save must not fail
if the card can't be built. The exception handler is covered by code
inspection, not asserted here.
"""

import os

import torch
from torch import nn

from src.train.snapper import Snapper


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def _load_card(snapshot_path):
    snap = torch.load(snapshot_path, map_location='cpu', weights_only=False)
    return snap['MODEL_CARD']


def test_save_embeds_card_inside_pt(tmp_snapshot_dir):
    """The card is in-band (no sibling .yaml any more)."""
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=2, _step=42, _async=False)

    pt = os.path.join(tmp_snapshot_dir, "002-0042.pt")
    yml = os.path.join(tmp_snapshot_dir, "002-0042.yaml")
    assert os.path.exists(pt), "model checkpoint missing"
    # The sibling .yaml is gone — card lives inside the .pt now.
    assert not os.path.exists(yml), "stale sibling yaml should not be written"

    card = _load_card(pt)
    assert card['snapshot_filename'] == "002-0042.pt"
    assert card['epoch'] == 2
    assert card['step'] == 42


def test_model_card_records_expected_fields(tmp_snapshot_dir):
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=0, _step=1, _async=False)

    card = _load_card(os.path.join(tmp_snapshot_dir, "000-0001.pt"))

    # Snapshot identity
    assert card['snapshot_filename'] == "000-0001.pt"
    assert card['epoch'] == 0
    assert card['step'] == 1
    assert 'seen_labels' not in card  # concept removed

    # Provenance (presence — exact values are runtime-dependent).
    assert 'created_utc' in card
    assert 'torch_version' in card and card['torch_version']
    assert 'python_version' in card and card['python_version']
    assert 'torch_initial_seed' in card

    # cuda_version may be None on CPU-only builds; just check the key exists.
    assert 'cuda_version' in card
    assert 'gpu_name' in card

    # git_sha is best-effort; key present even if value is None.
    assert 'git_sha' in card


def test_model_card_records_iso_timestamp(tmp_snapshot_dir):
    """The created_utc must be ISO-8601 (parseable by datetime.fromisoformat)."""
    from datetime import datetime

    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=0, _step=1, _async=False)

    card = _load_card(os.path.join(tmp_snapshot_dir, "000-0001.pt"))
    datetime.fromisoformat(card['created_utc'])


def test_load_still_works_after_card_writing(tmp_snapshot_dir):
    """The card-writing path must not break the basic save/load roundtrip."""
    src = _TinyModel()
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(src, _epoch=0, _step=1, _async=False)

    dst = _TinyModel()
    info = snapper.load(dst, _device='cpu',
                        _path=os.path.join(tmp_snapshot_dir, "000-0001.pt"))
    assert info is not None
    assert info['epoch'] == 0
    assert info['step'] == 1
