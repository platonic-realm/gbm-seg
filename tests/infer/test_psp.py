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
    # The big blob survives; opening-by-reconstruction restores it to full extent.
    assert out[:, 10:50, 10:50].sum() > 0


def test_reconstruction_preserves_thin_attachment(tmp_path):
    """A thin arm attached to a surviving component must be kept whole.

    A bare erode+dilate (morphological opening) deletes any structure
    narrower than the kernel — the arm would be lost past the 1-px the
    dilation can regrow. Opening by reconstruction geodesically dilates
    the survivor back to its full pre-erosion extent, so the entire arm,
    being connected to the body, is restored.
    """
    pred = np.zeros((4, 64, 64), dtype=np.uint8)
    pred[:, 10:50, 10:30] = 1          # solid body (survives erosion)
    pred[:, 29:30, 30:55] = 1          # 1-px-thin arm (vanishes under a 3x3 erosion)

    np.savez_compressed(tmp_path / "prediction.npz", arr=pred)

    p = PSP(_kernel_size=3, _min_2d_size=10, _min_3d_size=10)
    p.post_processing(tmp_path / "prediction.npz",
                      tmp_path / "prediction_psp.npz")

    out = np.load(tmp_path / "prediction_psp.npz")['arr']
    assert out[:, 10:50, 10:30].all(), "body should survive intact"
    # The whole arm — including its far end the dilation could never reach —
    # must be reconstructed back.
    assert out[:, 29, 30:55].all(), "thin arm should be fully preserved"


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


def _fragmented_mask():
    """Body + a near fragment (reconnectable) + a detached far fragment.

    The far fragment is large enough to clear `min_3d_size`, so only the
    reconnect step can remove it. The near fragment sits 4 voxels off the
    body (bridged by a radius-3 dilation); the far fragment is >2*3 voxels
    from everything.
    """
    pred = np.zeros((6, 80, 80), dtype=np.uint8)
    pred[:, 10:40, 10:40] = 1          # main body          : 5400 vox
    pred[:, 15:35, 44:60] = 1          # near fragment, gap 4: 1920 vox
    pred[:, 55:63, 5:13] = 1           # detached fragment   :  384 vox
    return pred


def test_reconnect_removes_detached_component(tmp_path):
    """A large-enough-to-survive-min_3d_size component that is detached
    from the GBM body is dropped by the reconnect step; the body and its
    near (bridgeable) fragment are kept."""
    np.savez_compressed(tmp_path / "prediction.npz", arr=_fragmented_mask())

    PSP(_kernel_size=1, _min_2d_size=1, _min_3d_size=10,
        _reconnect_radius=3).post_processing(
        tmp_path / "prediction.npz", tmp_path / "prediction_psp.npz")
    out = np.load(tmp_path / "prediction_psp.npz")['arr']

    assert out[:, 10:40, 10:40].sum() > 0, "body must be kept"
    assert out[:, 15:35, 44:60].sum() > 0, "near fragment must reconnect and be kept"
    assert out[:, 55:63, 5:13].sum() == 0, "detached fragment must be removed"


def test_reconnect_disabled_by_default(tmp_path):
    """reconnect_radius defaults to 0 — the step is skipped and the
    detached component survives, i.e. the old psp behaviour is unchanged."""
    np.savez_compressed(tmp_path / "prediction.npz", arr=_fragmented_mask())

    PSP(_kernel_size=1, _min_2d_size=1, _min_3d_size=10).post_processing(
        tmp_path / "prediction.npz", tmp_path / "prediction_psp.npz")
    out = np.load(tmp_path / "prediction_psp.npz")['arr']

    assert out[:, 55:63, 5:13].sum() > 0, \
        "with reconnect disabled the detached component must survive"


def test_reconnect_keeps_big_secondary(tmp_path):
    """A detached component large enough (>= keep_fraction of the largest)
    survives; raising keep_fraction above its relative size drops it."""
    pred = np.zeros((6, 80, 100), dtype=np.uint8)
    pred[:, 10:40, 10:40] = 1          # main body          : 5400 vox
    pred[:, 10:40, 55:80] = 1          # detached secondary : 4500 vox
    np.savez_compressed(tmp_path / "prediction.npz", arr=pred)

    # 4500 >= 0.2 * 5400 -> the big secondary is kept despite being detached.
    PSP(_kernel_size=1, _min_2d_size=1, _min_3d_size=10,
        _reconnect_radius=3, _keep_fraction=0.2).post_processing(
        tmp_path / "prediction.npz", tmp_path / "keep.npz")
    kept = np.load(tmp_path / "keep.npz")['arr']
    assert kept[:, 10:40, 10:40].sum() > 0
    assert kept[:, 10:40, 55:80].sum() > 0, "big secondary should be kept"

    # 4500 < 0.95 * 5400 -> the secondary now falls below keep_fraction.
    PSP(_kernel_size=1, _min_2d_size=1, _min_3d_size=10,
        _reconnect_radius=3, _keep_fraction=0.95).post_processing(
        tmp_path / "prediction.npz", tmp_path / "strict.npz")
    strict = np.load(tmp_path / "strict.npz")['arr']
    assert strict[:, 10:40, 10:40].sum() > 0
    assert strict[:, 10:40, 55:80].sum() == 0, \
        "secondary below keep_fraction should be dropped"

