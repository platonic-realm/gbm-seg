# Python Imports
import logging
import multiprocessing
import os
from pathlib import Path

# Library Imports
import imageio
import numpy as np
from skimage import measure, morphology

# Local Imports


class PSP:
    """Per-sample post-processing of a binary GBM mask.

    3D pass:  (optional) fill small enclosed cavities -> drop 3D connected
              components below ``min_3d_size`` voxels.
    2D pass (per Z-slice): (optional) fill small holes -> erode -> drop 2D
              components below ``min_2d_size`` px -> *opening by
              reconstruction*.

    Why reconstruction instead of a plain dilation: a bare erode+dilate is
    a morphological opening, which is anti-extensive — it can only remove
    voxels and systematically thins anything narrower than the kernel.
    The GBM is a thin membrane and ``prediction_psp.npz`` is the input to
    the thickness morphometry (``misc.morph_analysis``), so an opening
    here would bias the headline measurement. Opening by reconstruction
    instead geodesically dilates the surviving components back to their
    exact pre-erosion extent: components that vanish under erosion
    (specks, thin noise) are dropped, every survivor is left untouched.

    Full connectivity is used for every labelling step. The GBM is a thin
    curved sheet that meets the pixel grid diagonally everywhere; face-only
    connectivity would fragment one real arc into many small pieces that
    then fall below the size threshold and get deleted.

    Hole-filling is opt-in (``max_*_hole_size`` default 0 = disabled): a
    threshold must stay well below a capillary-lumen cross-section so a
    real lumen is never filled.
    """

    def __init__(self,
                 _kernel_size: int,
                 _min_2d_size: int,
                 _min_3d_size: int,
                 _max_2d_hole_size: int = 0,
                 _max_3d_hole_size: int = 0):

        self.kernel_size: int = _kernel_size
        self.min_2d_size: int = _min_2d_size
        self.min_3d_size: int = _min_3d_size
        self.max_2d_hole_size: int = _max_2d_hole_size
        self.max_3d_hole_size: int = _max_3d_hole_size

    def parallel_post_processing(self,
                                 _results_path: str,
                                 _max_concurrent: int) -> None:
        """Run ``post_processing`` on every sample dir under ``_results_path``."""

        inference_dir = Path(_results_path)
        sample_dirs = [d for d in inference_dir.iterdir() if d.is_dir()]

        tasks = []

        logging.info("Generating post processsing tasks")
        for sample in sample_dirs:
            input_path = sample / "prediction.npz"
            output_path = sample / "prediction_psp.npz"

            tasks.append((input_path,
                          output_path))

        logging.info("No of tasks: %d", len(tasks))

        logging.info("Creating a mulitprocessing pool with %d processes", _max_concurrent)
        with multiprocessing.Pool(_max_concurrent) as pool:
            pool.starmap(self.post_processing, tasks)

    def post_processing(self,
                        _input_path: Path,
                        _output_path: Path,
                        _multiplier: int = 120) -> None:

        PID = os.getpid()
        logging.info("Process %d: started!", PID)

        mask = np.load(_input_path)['arr'].astype(bool)
        logging.info("Process %d: prediction loaded, shape=%s", PID, mask.shape)

        # --- 3D pass: optional cavity fill, then drop small components ---
        if self.max_3d_hole_size > 0:
            mask = morphology.remove_small_holes(
                mask, area_threshold=self.max_3d_hole_size, connectivity=3)
        logging.info("Process %d: removing 3D objects smaller than %d",
                     PID, self.min_3d_size)
        mask = morphology.remove_small_objects(
            mask, min_size=self.min_3d_size, connectivity=3)
        logging.info("Process %d: 3D post processing finished", PID)

        # --- 2D pass: per-slice opening by reconstruction ---
        logging.info("Process %d: removing 2D objects smaller than %d",
                     PID, self.min_2d_size)
        kernel = morphology.rectangle(self.kernel_size, self.kernel_size)
        for i in range(mask.shape[0]):
            sample = mask[i]
            if self.max_2d_hole_size > 0:
                sample = morphology.remove_small_holes(
                    sample, area_threshold=self.max_2d_hole_size,
                    connectivity=2)

            # Erosion breaks thin noise-bridges, so a noise blob bridged to
            # real structure becomes a separate component and can be
            # dropped on its own. The size threshold is read off the
            # eroded slice; survivors are restored below by reconstruction.
            eroded = morphology.erosion(sample, kernel)
            labels = measure.label(eroded, connectivity=2)
            counts = np.bincount(labels.ravel())
            keep = counts >= self.min_2d_size
            keep[0] = False  # label 0 is background
            seed = keep[labels]

            if seed.any():
                # Geodesic dilation of the survivors, clipped to `sample`:
                # grows each survivor back to its full pre-erosion extent
                # while the dropped components (absent from the seed) stay
                # gone. Cast to uint8 — reconstruction expects a numeric
                # intensity image, not bool.
                mask[i] = morphology.reconstruction(
                    seed.astype(np.uint8),
                    sample.astype(np.uint8),
                    method='dilation').astype(bool)
            else:
                mask[i] = False

        logging.info("Process %d: 2D post processing finished", PID)

        prediction = mask.astype(np.uint8)
        np.savez_compressed(_output_path, arr=prediction)
        logging.info("Process %d: processed prediction numpy array saved", PID)

        gif_path = _output_path.parent / "prediction_psp.gif"

        # Scale 0/1 -> 0/_multiplier so the mask is visible in the gif.
        frames = (prediction * _multiplier).astype(np.uint8)
        with imageio.get_writer(gif_path, mode='I') as writer:
            for index in range(frames.shape[0]):
                writer.append_data(frames[index])

        logging.info("Process %d: processed prediction gif saved", PID)
