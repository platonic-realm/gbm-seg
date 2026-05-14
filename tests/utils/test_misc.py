"""Tests for src/utils/misc.py — loader guard + resize/voxel-size guard."""

import textwrap
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.utils.misc import (
    REMOVED_CONFIG_KEYS,
    _check_removed_keys,
    _has_dotted_key,
    get_voxel_size,
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


# --- get_voxel_size: fail loud on missing X/Y resolution -------------------


class _FakePage:
    """Minimal stand-in for tifffile.TiffPage exposing only `.tags` (a dict)."""

    def __init__(self, tags):
        self.tags = tags


class _FakeTiff:
    """Minimal stand-in for tifffile.TiffFile for get_voxel_size."""

    def __init__(self, tags=None, imagej_metadata=None):
        self.pages = [_FakePage(tags or {})]
        self.imagej_metadata = imagej_metadata


def test_get_voxel_size_raises_when_xy_resolution_missing_strict():
    """Strict mode (default): raise when X/YResolution missing.

    Pre-fix, the silent fallback to 1.0 µm/pixel combined with the
    default_voxel_size=[0.05, ...] target produced a 5% shrink (2048→102).
    With _default=None we now refuse to fall back at all. Mock TiffFile
    surface directly because tifffile.imwrite auto-emits defaults.
    """
    fake = _FakeTiff(tags={}, imagej_metadata={'ImageJ': '1.54f'})
    with pytest.raises(ValueError, match=r"[XY]Resolution"):
        get_voxel_size(fake, _path="bad.tiff")


def test_get_voxel_size_soft_fallback_when_xy_missing(caplog):
    """Soft mode (resize_and_copy): supply _default → fall back, warn."""
    fake = _FakeTiff(tags={}, imagej_metadata={'ImageJ': '1.54f'})
    target = [0.05, 0.05, 0.3]
    import logging as _logging
    with caplog.at_level(_logging.WARNING):
        x, y, z = get_voxel_size(fake, _path="bad.tiff", _default=target)
    assert x == 0.05
    assert y == 0.05
    # zoom_factor = target / voxel_size = 1.0 → no shrink.
    assert any("bad.tiff" in r.message and "Resolution" in r.message
               for r in caplog.records)


def test_get_voxel_size_succeeds_with_resolution(tmp_path):
    arr = np.zeros((4, 4, 32, 32), dtype=np.float32)
    good = tmp_path / "with_resolution.tiff"
    # 20 px / µm → voxel size = 0.05 µm/pixel.
    tifffile.imwrite(good, arr, shape=arr.shape, imagej=True,
                     resolution=(20.0, 20.0))

    with tifffile.TiffFile(good) as t:
        x, y, z = get_voxel_size(t, _path=good)
    assert abs(x - 0.05) < 1e-6
    assert abs(y - 0.05) < 1e-6
