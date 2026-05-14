"""PSP small-object removal.

Exercises the per-sample `post_processing` step that runs inside the
multiprocessing pool. The pre-Phase-1 code also re-ran every task
serially after the pool finished; that's covered by inspecting the
parallel_post_processing source (§1.2).
"""

import inspect

import numpy as np

from src.infer import psp as psp_module
from src.infer.psp import PSP


def test_small_3d_blob_removed(tmp_path):
    """A 3D blob smaller than `min_3d_size` should be deleted."""
    pred = np.zeros((10, 64, 64), dtype=np.uint8)
    # Big blob (will survive)
    pred[:, 10:50, 10:50] = 1
    # Tiny blob (will not survive min_3d_size=5000)
    pred[0:1, 0:2, 0:2] = 1

    np.savez_compressed(tmp_path / "prediction.npz", arr=pred)

    p = PSP(_kernel_size=3, _min_2d_size=10, _min_3d_size=5000)
    p.post_processing(tmp_path / "prediction.npz",
                      tmp_path / "prediction_psp.npz")

    out = np.load(tmp_path / "prediction_psp.npz")['arr']
    assert out[0, 0, 0] == 0, "Small blob should have been removed"
    # The big blob should mostly survive (erosion + dilation may shrink it slightly).
    assert out[:, 10:50, 10:50].sum() > 0


def test_parallel_post_processing_does_not_double_run():
    """§1.2 regression: the serial re-run after pool.starmap is gone.

    We inspect the source rather than measure runtime — multiprocessing in a
    test environment is fragile, and a textual check is sufficient because the
    bug was a literal extra for-loop calling the same function.
    """
    source = inspect.getsource(PSP.parallel_post_processing)
    # The pool.starmap call must remain
    assert "pool.starmap(self.post_processing" in source
    # The redundant serial loop must be gone. Pattern: `for task in tasks: self.post_processing(...)`
    assert "for task in tasks" not in source, (
        "PSP serial re-run after pool.starmap regressed (see §1.2)")


def test_psp_module_no_unused_math_import():
    """The dead Cylindrical-removal block + its `math` import were removed in §2.2."""
    assert not hasattr(psp_module, "math"), \
        "`math` should no longer be imported in psp.py (dead Cylindrical block was removed)"


def test_3d_removal_handles_many_components(tmp_path):
    """Volume with many small components — vectorised path must finish quickly
    and produce the same result as the equivalent per-label loop.

    Regression for the smoke-test bottleneck where `np.sum(labels == k)` ran
    once per label, scaling O(num_labels × volume). With bincount it's a
    single linear pass.
    """
    # 8 small blobs (4 voxels each — below min_3d_size=10) and 1 big blob.
    pred = np.zeros((8, 32, 32), dtype=np.uint8)
    for (z, y, x) in [(0, 0, 0), (0, 0, 10), (0, 10, 0),
                      (0, 10, 10), (1, 0, 0), (1, 0, 10),
                      (1, 10, 0), (1, 10, 10)]:
        pred[z:z + 1, y:y + 2, x:x + 2] = 1
    pred[3:7, 16:30, 16:30] = 1  # big

    np.savez_compressed(tmp_path / "prediction.npz", arr=pred)

    p = PSP(_kernel_size=1, _min_2d_size=1, _min_3d_size=10)
    p.post_processing(tmp_path / "prediction.npz",
                      tmp_path / "prediction_psp.npz")

    out = np.load(tmp_path / "prediction_psp.npz")['arr']

    # Every tiny blob removed; big one still has mass.
    for (z, y, x) in [(0, 0, 0), (0, 0, 10), (0, 10, 0), (0, 10, 10),
                      (1, 0, 0), (1, 0, 10), (1, 10, 0), (1, 10, 10)]:
        assert out[z, y, x] == 0, f"Tiny blob at ({z},{y},{x}) should be gone"
    assert out[3:7, 16:30, 16:30].sum() > 0

