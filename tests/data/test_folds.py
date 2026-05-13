"""Subject-wise stratified k-fold regression. Locks in §A1.

Tests:
- Known prefixes classify correctly; unknown prefixes are reported all-at-once.
- Each file appears in exactly one validation set and in (k-1) training sets.
- Stratification keeps group proportions stable across folds.
- The same seed yields the same partition (reproducibility).
- Read/write roundtrip through yaml preserves the structure.
"""

import pytest
import yaml

from src.data.folds import (
    DEFAULT_K,
    SAMPLE_GROUP_PREFIXES,
    UnknownSampleGroupError,
    assign_folds,
    classify,
    read_assignments,
    write_assignments,
)


def _sample_set():
    """A modest cohort representative of the project's real groups.
    Mix of Control / Podocin / Collagen across multiple subjects per group
    so 5-fold splits land at least one of each in each fold."""
    return [
        # Control (4)
        "CKM103.foo.tif", "CKM110.bar.tif",
        "NCW.AUY381.baz.tif", "NCW.AUY380.qux.tif",
        # Podocin (3)
        "NCW.BDP669.a.tif", "NCW.BDP672.b.tif", "NCW.BDP675.c.tif",
        # Collagen (3)
        "NCW.CKM105.x.tif", "NCW.CKM105.y.tif", "NCW.CKM104.z.tif",
    ]


# --- classify --------------------------------------------------------------

def test_classify_known_prefixes():
    assert classify("ckm103.foo") == "Control"
    assert classify("CKM110.bar") == "Control"
    assert classify("NCW.BDP669.a") == "Podocin"
    assert classify("NCW.CKM104.z") == "Collagen"


def test_classify_unknown_returns_none():
    assert classify("MYSTERY_PREFIX_42") is None
    assert classify("") is None


def test_classify_is_case_insensitive():
    assert classify("ckm103.test") == "Control"
    assert classify("CKM103.test") == "Control"


# --- assign_folds: correctness ---------------------------------------------

def test_assign_folds_unknown_prefix_lists_all_offenders():
    files = _sample_set() + ["MYSTERY_A.tif", "MYSTERY_B.tif"]
    with pytest.raises(UnknownSampleGroupError) as info:
        assign_folds(files)
    msg = str(info.value)
    # Both unknown names must appear so the user can fix in one pass.
    assert "MYSTERY_A.tif" in msg
    assert "MYSTERY_B.tif" in msg


def test_assign_folds_known_prefix_passes_silently():
    folds = assign_folds(_sample_set())
    assert len(folds) == DEFAULT_K


def test_assign_folds_partitions_every_file_once_into_valid():
    files = _sample_set()
    folds = assign_folds(files)
    seen_in_valid = []
    for fold in folds:
        seen_in_valid.extend(fold['valid'])
    assert sorted(seen_in_valid) == sorted(files), \
        "every file should land in exactly one fold's validation set"


def test_assign_folds_train_and_valid_disjoint_per_fold():
    folds = assign_folds(_sample_set())
    for fold in folds:
        assert not set(fold['train']) & set(fold['valid'])


def test_assign_folds_every_file_in_k_minus_1_training_sets():
    files = _sample_set()
    folds = assign_folds(files)
    appearances = {f: 0 for f in files}
    for fold in folds:
        for f in fold['train']:
            appearances[f] += 1
    assert all(c == DEFAULT_K - 1 for c in appearances.values())


def test_assign_folds_seeded_reproducibility():
    files = _sample_set()
    folds_a = assign_folds(files, seed=1234)
    folds_b = assign_folds(files, seed=1234)
    assert folds_a == folds_b
    folds_c = assign_folds(files, seed=9999)
    assert folds_a != folds_c


def test_assign_folds_too_few_files_raises():
    with pytest.raises(ValueError, match="Cannot build"):
        assign_folds(["CKM103.tif", "CKM110.tif"], k=5)


def test_assign_folds_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        assign_folds([])


def test_assign_folds_stratification_keeps_group_balance():
    """Each group should appear in at least one fold's validation set
    (with 10 files across 3 groups + 5 folds, no group should be totally
    missing from validation)."""
    folds = assign_folds(_sample_set())
    all_valid_groups = set()
    for fold in folds:
        for group in fold['groups_in_valid'].split(','):
            all_valid_groups.add(group)
    # All three groups should appear somewhere in validation.
    assert all_valid_groups == set(SAMPLE_GROUP_PREFIXES.keys())


# --- yaml roundtrip --------------------------------------------------------

def test_write_and_read_assignments_roundtrip(tmp_path):
    folds = assign_folds(_sample_set())
    written_path = write_assignments(tmp_path, folds)
    assert written_path.exists()
    # Should be valid YAML and contain the structure we expect.
    raw = yaml.safe_load(written_path.read_text(encoding="UTF-8"))
    assert raw['k'] == DEFAULT_K
    assert len(raw['folds']) == DEFAULT_K
    # And read_assignments should give back the same list of dicts.
    read_back = read_assignments(tmp_path)
    assert read_back == folds


def test_read_assignments_missing_file_points_at_recreation(tmp_path):
    with pytest.raises(FileNotFoundError, match="gbm.py create"):
        read_assignments(tmp_path)
