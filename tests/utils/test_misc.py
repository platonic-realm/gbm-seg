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
    _z_interpolate_and_stack,
    get_voxel_size,
    resize_and_copy,
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
        "trainer.logging.visualization.blender",
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


# --- _z_interpolate_and_stack: restore of the deleted interpolate.py logic --


def test_z_interp_no_op_for_scale_one():
    img = np.random.default_rng(0).standard_normal((4, 4, 8, 8)).astype(np.float32)
    out = _z_interpolate_and_stack(img, 1, _has_labels=True)
    # Same array returned unchanged.
    assert out is img


def test_z_interp_stacks_labels_by_repeat():
    """Labels (channel 3) must be `np.repeat`'d along Z, not interpolated."""
    z, c, h, w = 3, 4, 4, 4
    img = np.zeros((z, c, h, w), dtype=np.float32)
    # Distinct label per source Z so we can verify the replicate pattern.
    img[0, 3] = 10
    img[1, 3] = 20
    img[2, 3] = 30

    out = _z_interpolate_and_stack(img, 6, _has_labels=True)
    assert out.shape == (z * 6, c, h, w)

    label_z = out[:, 3, 0, 0]
    expected = np.array([10] * 6 + [20] * 6 + [30] * 6, dtype=np.float32)
    np.testing.assert_array_equal(label_z, expected)


def test_z_interp_uses_trilinear_for_intensity_channels():
    """Linear-ish gradient in Z should remain linear-ish in the upsampled output."""
    z, h, w = 4, 4, 4
    img = np.zeros((z, 4, h, w), dtype=np.float32)
    # Channel 0 has a clean Z gradient 0, 10, 20, 30; channels 1-2 mirror;
    # channel 3 is labels (zeros, irrelevant here).
    for i in range(z):
        img[i, 0] = i * 10.0
        img[i, 1] = i * 10.0
        img[i, 2] = i * 10.0

    out = _z_interpolate_and_stack(img, 4, _has_labels=True)
    assert out.shape == (z * 4, 4, h, w)
    # Interpolation should bracket the source values: min and max preserved.
    assert out[:, 0, 0, 0].min() >= 0.0
    assert out[:, 0, 0, 0].max() <= 30.0
    # Should be monotonically non-decreasing for a strictly increasing source.
    diffs = np.diff(out[:, 0, 0, 0])
    assert (diffs >= -1e-5).all(), "trilinear upsample should be non-decreasing"


def test_resize_and_copy_z_scale_pipeline(tmp_path):
    """End-to-end: a TIFF copied with z_scale=3 has 3x the Z slices, and
    labels replicate every 3rd slice. Also confirm `resolution=` round-trips
    so get_voxel_size doesn't fail (the smoke-time concern)."""
    src = tmp_path / "ds_src"
    src.mkdir()
    # The threshold inside `resize_and_copy` snaps labels to {0, 255}, so
    # distinguish source slices by alternating those two values.
    arr = np.zeros((4, 4, 32, 32), dtype=np.float32)
    for i in range(4):
        arr[i, :3] = i * 50.0
        arr[i, 3] = 255.0 if i % 2 == 1 else 0.0
    tifffile.imwrite(src / "tiny.tiff", arr,
                     shape=arr.shape, imagej=True,
                     resolution=(20.0, 20.0))

    dest = tmp_path / "ds_dest"
    dest.mkdir()
    resize_and_copy(str(src), str(dest),
                    _target_size=[0.05, 0.05, 0.3],
                    _z_scale_factor=3)

    out_files = sorted(Path(dest).glob("*.tiff"))
    assert len(out_files) == 1
    out_img = tifffile.imread(out_files[0])
    assert out_img.shape == (12, 4, 32, 32)
    # Labels: np.repeat'd; original slice i sits at out Z = 3i..3i+2.
    label_z = out_img[:, 3, 0, 0]
    expected = np.repeat(
        np.array([0.0, 255.0, 0.0, 255.0], dtype=np.float32), 3)
    np.testing.assert_array_equal(label_z, expected)


def test_resize_and_copy_parallel_matches_serial(tmp_path):
    """The multiprocessing Pool path must produce exactly the same resized
    volumes as the serial path — parallelism is an optimisation, never a
    behaviour change. Volumes of differing Z-depth give the workers uneven
    tasks so out-of-order completion is genuinely exercised."""
    src = tmp_path / "ds_src"
    src.mkdir()
    rng = np.random.default_rng(0)
    for idx, depth in enumerate((3, 5, 4, 2)):
        arr = np.zeros((depth, 4, 24, 24), dtype=np.float32)
        for z in range(depth):
            arr[z, :3] = rng.integers(0, 200, size=(3, 24, 24))
            arr[z, 3] = 255.0 if z % 2 else 0.0
        tifffile.imwrite(src / f"vol_{idx}.tiff", arr,
                         shape=arr.shape, imagej=True,
                         resolution=(20.0, 20.0))

    dest_serial = tmp_path / "serial"
    dest_parallel = tmp_path / "parallel"
    dest_serial.mkdir()
    dest_parallel.mkdir()

    resize_and_copy(str(src), str(dest_serial),
                    _target_size=[0.05, 0.05, 0.3],
                    _z_scale_factor=3, _workers=1)
    resize_and_copy(str(src), str(dest_parallel),
                    _target_size=[0.05, 0.05, 0.3],
                    _z_scale_factor=3, _workers=4)

    serial_files = sorted(p.name for p in dest_serial.glob("*.tiff"))
    parallel_files = sorted(p.name for p in dest_parallel.glob("*.tiff"))
    assert serial_files == parallel_files
    assert len(serial_files) == 4
    for name in serial_files:
        a = tifffile.imread(dest_serial / name)
        b = tifffile.imread(dest_parallel / name)
        np.testing.assert_array_equal(
            a, b, err_msg=f"{name} differs between serial and parallel resize")


def test_resize_and_copy_empty_source_is_noop(tmp_path):
    """A source dir with no TIFFs must not raise (e.g. an absent optional
    split) — the Pool branch should never be entered on an empty task list."""
    src = tmp_path / "empty"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    resize_and_copy(str(src), str(dest), _target_size=[0.05, 0.05, 0.3],
                    _z_scale_factor=3)
    assert list(dest.iterdir()) == []


def test_create_new_experiment_three_way_dataset_split(tmp_path):
    """create_new_experiment produces a 3-way dataset split:

    - ds_train          — Z-upsampled by z_scale at create time.
    - ds_test_unlabeled — flat whole-glom volumes, native Z (inference
                          does the Z upsampling on-the-fly; pre-upsampling
                          here would double the inflation).
    - ds_test_labeled   — one sub-directory per annotator, native Z.

    Also asserts the per-experiment configs.yaml routes the inference
    paths to the new directory names.
    """
    import yaml as _yaml

    from src.utils.exper import create_new_experiment

    src_root = tmp_path / "src_dataset"
    (src_root / "ds_train").mkdir(parents=True)
    (src_root / "ds_test_unlabeled").mkdir(parents=True)
    experts = ("Chris", "David", "Robin")
    for expert in experts:
        (src_root / "ds_test_labeled" / expert).mkdir(parents=True)

    # 4-channel (image + label) array for train + labeled crops.
    arr4 = np.zeros((4, 4, 32, 32), dtype=np.float32)
    arr4[:, :3] = 100.0
    # 3-channel (image only) array for the unlabeled whole-glom volumes.
    arr3 = arr4[:, :3].copy()

    # Filenames must match SAMPLE_GROUP_PREFIXES so assign_folds doesn't
    # bail. Five distinct train files cover the k=5 fold requirement.
    train_names = [
        "NCW.AUY381.fake1.tiff",
        "NCW.AUY380.fake2.tiff",
        "NCW.BDP669.fake3.tiff",
        "NCW.BDP672.fake4.tiff",
        "NCW.CKM104.fake5.tiff",
    ]
    for name in train_names:
        tifffile.imwrite(src_root / "ds_train" / name, arr4,
                         shape=arr4.shape, imagej=True, resolution=(20.0, 20.0))
    tifffile.imwrite(src_root / "ds_test_unlabeled" / "NCW.CKM105.whole.tiff",
                     arr3, shape=arr3.shape, imagej=True,
                     resolution=(20.0, 20.0))
    for expert in experts:
        tifffile.imwrite(
            src_root / "ds_test_labeled" / expert / "NCW.CKM105.crop.tiff",
            arr4, shape=arr4.shape, imagej=True, resolution=(20.0, 20.0))

    exp_root = tmp_path / "experiments"
    exp_root.mkdir()

    # `create_new_experiment` writes git_sha / pip freeze / a real source
    # snapshot — point it at a temp _source_path with just the configs/
    # template so it can finish without exploding.
    src_code = tmp_path / "source"
    (src_code / "configs").mkdir(parents=True)
    # Minimal template with the fields `create_new_experiment` reads.
    (src_code / "configs" / "template.yaml").write_text(
        "experiments:\n  root: ./\n  default_data_path: ./\n  "
        "default_batch_size: 8\n  default_voxel_size: [0.05, 0.05, 0.3]\n  "
        "scale_learning_rate_for_batch_size: True\n"
        "trainer:\n"
        "  optimization: {epochs: 1, optim: {lr: 0.0001}}\n"
        "  data:\n"
        "    train_ds: {path: '', batch_size: 8, augmentation: {}}\n"
        "    valid_ds: {path: '', batch_size: 8}\n"
        "  logging: {report_freq: 1000}\n"
        "inference:\n  inference_ds: {path: '', batch_size: 8}\n"
        "  labeled_test_ds: {path: ''}\n"
        "root_path: ./\n")
    # The function reads from cwd-relative ./configs/template.yaml, so run
    # from src_code.
    import os as _os
    cwd_before = _os.getcwd()
    try:
        _os.chdir(src_code)
        create_new_experiment(
            _name="mini", _root_path=str(exp_root), _source_path=str(src_code),
            _dataset_path=str(src_root), _batch_size=8,
            _voxel_size=[0.05, 0.05, 0.3], _z_scale_factor=6)
    finally:
        _os.chdir(cwd_before)

    ds = exp_root / "mini" / "datasets"
    train_out = tifffile.imread(ds / "ds_train" / "NCW.AUY381.fake1.tiff")
    unlabeled_out = tifffile.imread(
        ds / "ds_test_unlabeled" / "NCW.CKM105.whole.tiff")
    labeled_out = tifffile.imread(
        ds / "ds_test_labeled" / "Chris" / "NCW.CKM105.crop.tiff")

    # Train Z-upsampled 4 → 24 (× z_scale=6); both test sets stay native Z=4.
    assert train_out.shape[0] == 24, (
        f"ds_train must be Z-upsampled (got {train_out.shape})")
    assert unlabeled_out.shape[0] == 4, (
        f"ds_test_unlabeled must stay native Z (got {unlabeled_out.shape})")
    assert labeled_out.shape[0] == 4, (
        f"ds_test_labeled must stay native Z (got {labeled_out.shape})")

    # All three annotator sub-directories were copied.
    for expert in experts:
        assert (ds / "ds_test_labeled" / expert
                / "NCW.CKM105.crop.tiff").exists(), (
            f"missing labeled crops for annotator {expert}")

    # configs.yaml routes the inference paths to the new directory names.
    cfg = _yaml.safe_load((exp_root / "mini" / "configs.yaml").read_text())
    assert cfg['inference']['inference_ds']['path'].endswith(
        'ds_test_unlabeled/')
    assert cfg['inference']['labeled_test_ds']['path'].endswith(
        'ds_test_labeled/')


def test_copy_directory_survives_dangling_symlink(tmp_path):
    """copy_directory must not abort when the source tree contains a
    broken symlink. Regression: wandb writes `policy='live'` symlinks to
    snapshot files that may already be deleted; shutil.copytree's default
    raises on dangling links, which aborted the whole code/ snapshot at
    `gbm.py create` time."""
    from src.utils.misc import copy_directory

    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "real.txt").write_text("ok")
    # A dangling symlink — target never exists.
    (src / "sub" / "broken.pt").symlink_to(src / "sub" / "gone.pt")
    (src / "keep.py").write_text("code")

    dest = tmp_path / "dest"
    dest.mkdir()
    copy_directory(str(src), str(dest), ['.git'])

    # The copy completed despite the dangling link.
    assert (dest / "keep.py").read_text() == "code"
    assert (dest / "sub" / "real.txt").read_text() == "ok"


def test_copy_directory_honors_exclude_list(tmp_path):
    """Top-level items named in the exclude list are skipped entirely —
    this is how `gbm.py create` keeps wandb/ out of the code/ snapshot."""
    from src.utils.misc import copy_directory

    src = tmp_path / "src"
    (src / "wandb").mkdir(parents=True)
    (src / "wandb" / "junk.txt").write_text("x")
    (src / "code.py").write_text("y")

    dest = tmp_path / "dest"
    dest.mkdir()
    copy_directory(str(src), str(dest), ['wandb'])

    assert (dest / "code.py").exists()
    assert not (dest / "wandb").exists()


def test_copy_directory_skips_pycache(tmp_path):
    """__pycache__ / *.pyc are stripped at every nesting level."""
    from src.utils.misc import copy_directory

    src = tmp_path / "src"
    (src / "pkg" / "__pycache__").mkdir(parents=True)
    (src / "pkg" / "__pycache__" / "mod.cpython-39.pyc").write_text("bc")
    (src / "pkg" / "mod.py").write_text("src")

    dest = tmp_path / "dest"
    dest.mkdir()
    copy_directory(str(src), str(dest), [])

    assert (dest / "pkg" / "mod.py").exists()
    assert not (dest / "pkg" / "__pycache__").exists()
