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
