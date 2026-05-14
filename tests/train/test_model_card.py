"""Per-snapshot model-card regression. Locks in §E4.

When `Snapper.save` writes a `.pt` snapshot, it must also write a
sibling `.yaml` capturing snapshot-time provenance: epoch/step,
torch + CUDA versions, python version, torch.initial_seed(), git SHA
(best-effort), gpu name, creation timestamp.

The model card is a *best-effort* artifact: snapshot save must not fail
if the card can't be written (covered by the warning path, not asserted
here — it's an exception handler).
"""

import os

import yaml
from torch import nn

from src.train.snapper import Snapper


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_save_writes_yaml_sibling(tmp_snapshot_dir):
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=2, _step=42, _async=False)

    pt = os.path.join(tmp_snapshot_dir, "002-0042.pt")
    yml = os.path.join(tmp_snapshot_dir, "002-0042.yaml")
    assert os.path.exists(pt), "model checkpoint missing"
    assert os.path.exists(yml), "sibling model card missing"


def test_model_card_records_expected_fields(tmp_snapshot_dir):
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=0, _step=1, _async=False)

    with open(os.path.join(tmp_snapshot_dir, "000-0001.yaml"), encoding="UTF-8") as f:
        card = yaml.safe_load(f)

    # Snapshot identity fields
    assert card['snapshot_filename'] == "000-0001.pt"
    assert card['epoch'] == 0
    assert card['step'] == 1
    assert 'seen_labels' not in card  # concept removed

    # Provenance: timestamp, library versions, python version. We only assert
    # presence (not exact values), since they depend on the runtime.
    assert 'created_utc' in card
    assert 'torch_version' in card and card['torch_version']
    assert 'python_version' in card and card['python_version']
    assert 'torch_initial_seed' in card

    # cuda_version may legitimately be None on CPU-only builds; just check the key exists.
    assert 'cuda_version' in card
    assert 'gpu_name' in card

    # git_sha is best-effort; the *key* must be present even if the value is None.
    assert 'git_sha' in card


def test_model_card_records_iso_timestamp(tmp_snapshot_dir):
    """The created_utc must be ISO-8601 (parseable by datetime.fromisoformat)."""
    from datetime import datetime

    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(_TinyModel(), _epoch=0, _step=1, _async=False)

    with open(os.path.join(tmp_snapshot_dir, "000-0001.yaml"), encoding="UTF-8") as f:
        card = yaml.safe_load(f)

    # Should not raise
    datetime.fromisoformat(card['created_utc'])


def test_load_still_works_after_card_writing(tmp_snapshot_dir):
    """The card-writing path must not break the basic save/load roundtrip."""
    src = _TinyModel()
    snapper = Snapper(tmp_snapshot_dir)
    snapper.save(src, _epoch=0, _step=1, _async=False)

    dst = _TinyModel()
    result = snapper.load(dst, _device='cpu',
                          _path=os.path.join(tmp_snapshot_dir, "000-0001.pt"))
    assert result == 0  # epoch only
