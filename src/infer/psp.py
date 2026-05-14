# Python Imports
import logging
import multiprocessing
import os
from pathlib import Path

import imageio

# Library Imports
import numpy as np
from scipy import ndimage
from skimage import measure, morphology

# Local Imports


class PSP:
    """Per-sample post-processing: 2D erosion → connected-component filter by
    ``min_2d_size`` → dilation, then 3D connected-component filter by
    ``min_3d_size``. Parallelised across samples via multiprocessing.Pool.
    """

    def __init__(self,
                 _kernel_size: int,
                 _min_2d_size: int,
                 _min_3d_size: int):

        self.kernel_size: int = _kernel_size
        self.min_2d_size: int = _min_2d_size
        self.min_3d_size: int = _min_3d_size

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

        prediction = np.load(_input_path)['arr']
        logging.info("Process %d: prediction numpy array loaded", PID)

        labels, labels_num = ndimage.label(prediction)
        # Count voxels per label in a single linear pass via bincount, then
        # zero out every voxel whose label's count is below min_3d_size.
        # The pre-fix code did `np.sum(labels == k)` once per label — O(N×V)
        # — which pegged CPU for an hour on 60M+-voxel volumes.
        logging.info("Process %d: removing 3D objects smaller than %d "
                     "(labels=%d)", PID, self.min_3d_size, labels_num)
        if labels_num > 0:
            counts = np.bincount(labels.ravel())
            small_labels = np.where(counts < self.min_3d_size)[0]
            # Label 0 is background — it's always "small" in this sense but
            # already zero. Subtract a fast `labels == 0` test to avoid an
            # unnecessary isin pass on the background mask.
            small_labels = small_labels[small_labels != 0]
            if small_labels.size:
                prediction[np.isin(labels, small_labels)] = 0

        logging.info("Process %d: 3D post processing finished", PID)

        logging.info("Process %d: removing 2D objects smaller than %d", PID, self.min_2d_size)
        kernel = morphology.rectangle(self.kernel_size, self.kernel_size)
        for i in range(prediction.shape[0]):
            prediction[i, :, :] = morphology.erosion(prediction[i, :, :], kernel)

            sample = prediction[i, :, :]
            slice_labels = measure.label(sample, connectivity=1)
            # Same vectorisation as the 3D pass.
            unique_labels, label_counts = np.unique(slice_labels,
                                                    return_counts=True)
            small = unique_labels[(label_counts < self.min_2d_size)
                                  & (unique_labels != 0)]
            if small.size:
                prediction[i][np.isin(slice_labels, small)] = 0

            prediction[i, :, :] = morphology.dilation(prediction[i, :, :], kernel)

        logging.info("Process %d: 2D post processing finished", PID)

        np.savez_compressed(_output_path, arr=prediction)
        logging.info("Process %d: processed prediction numpy array saved", PID)

        gif_path = _output_path.parent / "prediction_psp.gif"

        prediction = prediction * _multiplier  # To differentiate the colors
        prediction = prediction.astype(np.uint8)

        with imageio.get_writer(gif_path, mode='I') as writer:
            for index in range(prediction.shape[0]):
                writer.append_data(prediction[index])

        logging.info("Process %d: processed prediction gif saved", PID)
