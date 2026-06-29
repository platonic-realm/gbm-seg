# Tests for the Z-continuity analysis (src/infer/continuity.py).

# Library
import numpy as np
import pytest
import yaml

# Local
from src.infer.continuity import (
    calculate_continuity,
    compare_continuity,
    continuity_metrics,
)


def _smooth_membrane(z=12, h=32, w=32):
    """A vertical slab occupying the same (h,w) block on EVERY Z slice —
    perfectly Z-continuous: no along-Z transitions inside the slab."""
    m = np.zeros((z, h, w), dtype=bool)
    m[:, 8:16, 8:24] = True
    return m


def _jagged_membrane(z=12, h=32, w=32):
    """The same block but present only on every OTHER Z slice — maximally
    jagged: every adjacent Z pair flips."""
    m = np.zeros((z, h, w), dtype=bool)
    m[::2, 8:16, 8:24] = True
    return m


def test_smooth_has_lower_ztv_than_jagged():
    smooth = continuity_metrics(_smooth_membrane())
    jagged = continuity_metrics(_jagged_membrane())
    assert smooth["z_tv_per_fg_voxel"] < jagged["z_tv_per_fg_voxel"]
    assert smooth["mean_flips_per_fg_column"] < jagged["mean_flips_per_fg_column"]
    assert smooth["mean_adjacent_iou"] > jagged["mean_adjacent_iou"]


def test_smooth_membrane_is_perfectly_continuous():
    m = continuity_metrics(_smooth_membrane())
    # A column present on every slice never flips along Z → 0 transitions.
    assert m["z_tv_per_fg_voxel"] == pytest.approx(0.0)
    assert m["mean_flips_per_fg_column"] == pytest.approx(0.0)
    # Consecutive slices are identical → adjacent IoU == 1.
    assert m["mean_adjacent_iou"] == pytest.approx(1.0)


def test_flip_count_matches_known_pattern():
    # Slab on slices {0,1,2}, absent on {3..}: each foreground column is
    # present for 3 slices then off → exactly ONE 1→0 flip along Z.
    m = np.zeros((6, 16, 16), dtype=bool)
    m[0:3, 4:8, 4:8] = True
    out = continuity_metrics(m)
    assert out["mean_flips_per_fg_column"] == pytest.approx(1.0)


def test_ztv_normalization_is_volume_fair():
    # Two slabs, one 2x the XY area of the other, both with the SAME jagged
    # every-other-slice pattern. Per-FG-voxel Z-TV must be equal (it's the
    # fairness control) even though raw transition counts differ.
    small = np.zeros((8, 32, 32), dtype=bool)
    small[::2, 0:8, 0:8] = True
    big = np.zeros((8, 32, 32), dtype=bool)
    big[::2, 0:16, 0:16] = True
    a = continuity_metrics(small)["z_tv_per_fg_voxel"]
    b = continuity_metrics(big)["z_tv_per_fg_voxel"]
    assert a == pytest.approx(b)


def test_empty_and_single_slice_are_nan_not_crash():
    empty = continuity_metrics(np.zeros((10, 8, 8), dtype=bool))
    assert empty["fg_voxels"] == 0
    assert np.isnan(empty["z_tv_per_fg_voxel"])
    single = continuity_metrics(np.ones((1, 8, 8), dtype=bool))
    assert single["z_slices"] == 1
    assert np.isnan(single["z_tv_per_fg_voxel"])


def _write_pred(sample_dir, fname, arr):
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_dir / fname, arr=arr.astype(np.uint8))


def test_calculate_continuity_end_to_end(tmp_path):
    infer = tmp_path / "tag"
    _write_pred(infer / "s1", "prediction.npz", _smooth_membrane())
    _write_pred(infer / "s1", "prediction_psp.npz", _smooth_membrane())
    _write_pred(infer / "s2", "prediction.npz", _jagged_membrane())
    _write_pred(infer / "s2", "prediction_psp.npz", _jagged_membrane())

    out_dir = tmp_path / "tag_continuity"
    res = calculate_continuity(infer, out_dir)

    assert res["n_samples"] == 2
    assert (out_dir / "continuity_result.yaml").exists()
    # both variants aggregated
    assert "raw" in res["aggregate"] and "psp" in res["aggregate"]
    assert res["aggregate"]["raw"]["z_tv_per_fg_voxel"]["n"] == 2
    # yaml round-trips
    loaded = yaml.safe_load((out_dir / "continuity_result.yaml").read_text())
    assert loaded["n_samples"] == 2


def test_calculate_continuity_skips_samples_without_predictions(tmp_path):
    infer = tmp_path / "tag"
    _write_pred(infer / "good", "prediction.npz", _smooth_membrane())
    (infer / "empty").mkdir(parents=True)   # no prediction files
    res = calculate_continuity(infer, tmp_path / "out")
    assert res["n_samples"] == 1


def test_compare_continuity_tabulates(tmp_path, caplog):
    # Build two minimal continuity_result.yaml files (cont smoother than CE).
    def mk(p, ztv):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump({
            "aggregate": {"raw": {
                "z_tv_per_fg_voxel": {"mean": ztv},
                "mean_flips_per_fg_column": {"mean": ztv * 2},
                "mean_adjacent_iou": {"mean": 1 - ztv},
                "fg_fraction": {"mean": 0.05}},
                "psp": {}}}))
    a = tmp_path / "cont.yaml"
    mk(a, 0.10)
    b = tmp_path / "ce.yaml"
    mk(b, 0.20)
    table = compare_continuity({"cont": a, "ce": b})
    assert table["raw"]["z_tv_per_fg_voxel"]["cont"] == pytest.approx(0.10)
    assert table["raw"]["z_tv_per_fg_voxel"]["ce"] == pytest.approx(0.20)
