"""Tests for src/utils/misc.py — specifically the removed-key loader guard."""

import textwrap
from pathlib import Path

import pytest

from src.utils.misc import (
    REMOVED_CONFIG_KEYS,
    _check_removed_keys,
    _has_dotted_key,
)


def test_has_dotted_key_finds_nested():
    cfg = {"a": {"b": {"c": 1}}}
    assert _has_dotted_key(cfg, "a.b.c")
    assert _has_dotted_key(cfg, "a.b")
    assert _has_dotted_key(cfg, "a")


def test_has_dotted_key_missing_branch():
    cfg = {"a": {"b": {"c": 1}}}
    assert not _has_dotted_key(cfg, "a.b.d")
    assert not _has_dotted_key(cfg, "x")
    assert not _has_dotted_key(cfg, "a.b.c.d")  # past leaf


def test_check_removed_keys_clean_config_passes():
    cfg = {"trainer": {"model": {"name": "unet_3d"}}}
    _check_removed_keys(cfg)  # no raise


def test_check_removed_keys_flags_tensorboard():
    cfg = {"trainer": {"tensorboard": {"enabled": False}}}
    with pytest.raises(ValueError, match=r"trainer\.tensorboard"):
        _check_removed_keys(cfg)


def test_check_removed_keys_flags_multiple():
    cfg = {
        "trainer": {
            "sqlite": False,
            "profiling": {"enabled": False},
        },
        "logging": {"log_levels": ["INFO"]},
    }
    with pytest.raises(ValueError) as exc:
        _check_removed_keys(cfg)
    msg = str(exc.value)
    assert "trainer.sqlite" in msg
    assert "trainer.profiling" in msg
    assert "logging.log_levels" in msg


def test_removed_config_keys_includes_known_dead_fields():
    for key in [
        "trainer.tensorboard",
        "trainer.sqlite",
        "trainer.profiling",
        "trainer.visualization.blender",
        "experiments.model_sizes",
        "experiments.train_same_sample_size",
    ]:
        assert key in REMOVED_CONFIG_KEYS


def test_read_configs_rejects_legacy_yaml(tmp_path: Path):
    """End-to-end: a YAML with a removed key raises through read_configs."""
    from src.utils.misc import read_configs

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(textwrap.dedent("""\
        trainer:
          tensorboard:
            enabled: false
            path: ./tb/
    """))

    with pytest.raises(ValueError, match=r"trainer\.tensorboard"):
        read_configs(str(legacy))
