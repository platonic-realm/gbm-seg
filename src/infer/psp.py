# Python Imports
import os
import logging
import multiprocessing
from pathlib import Path

# Library Imports
import numpy as np
import imageio
from skimage import measure, morphology
from scipy import ndimage

# Local Imports


class PSP():
    def __init__(self,
                 _kernel_size: int,
                 _min_2d_size: int,
                 _min_3d_size: int):

        self.kernel_size: int = _kernel_size
        self.min_2d_size: int = _min_2d_size
        self.min_3d_size: int = _min_3d_size

    def parallel_post_processing(self,
                                 _results_path: str,
                                 _max_concurrent: int):

        inference_dir = Path(_results_path)
        sample_dirs = [d for d in inference_dir.iterdir() if d.is_dir()]

        tasks = []

        logging.info("Generating post processsing tasks")
        for sample in sample_dirs:
            input_path = sample / "prediction.npy"
            output_path = sample / "prediction_psp.npy"

            tasks.append((input_path,
                          output_path))

        logging.info("No of tasks: %d", len(tasks))

        logging.info("Creating a mulitprocessing pool with %d processes", _max_concurrent)
        with multiprocessing.Pool(_max_concurrent) as pool:
            pool.starmap(self.post_processing, tasks)

    def post_processing(self,
                        _input_path: Path,
                        _output_path: Path,
                        _multipier: int = 120) -> None:

        PID = os.getpid()
        logging.info("Process %d: started!", PID)

        prediction = np.load(_input_path)
        logging.info("Process %d: prediction numpy array loaded", PID)

        for i in range(prediction.shape[0]):

            kernel = morphology.rectangle(self.kernel_size, self.kernel_size)
            eroded_image = morphology.erosion(prediction[i, :, :], kernel)
            prediction[i, :, :] = eroded_image

            sample = prediction[i, :, :]
            labels = measure.label(sample, connectivity=1)
            # count pixels in each connected component
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            # remove small connected components
            # print(f"mean: {int(np.mean(label_counts))}, std: {int(np.std(label_counts))}, max: {int(np.max(label_counts))}, min: {int(np.min(label_counts))}")
            logging.info("Process %d: removing 2D objects smaller than %d", PID, self.min_2d_size)
            for label, count in zip(unique_labels, label_counts):
                if count < self.min_2d_size and label != 0:
                    prediction[i, :, :][labels == label] = 0

            kernel = morphology.rectangle(self.kernel_size, self.kernel_size)
            dilated_image = morphology.dilation(prediction[i, :, :], kernel)
            prediction[i, :, :] = dilated_image

        logging.info("Process %d: 2D post processing finished", PID)

        labels, labels_num = ndimage.label(prediction)
        # count pixels in each connected component
        logging.info("Process %d: removing 3D objects smaller than %d", PID, self.min_3d_size)
        for label_index in range(labels_num):
            voxel_count = np.sum(labels == label_index)
            if voxel_count < self.min_3d_size:
                prediction[labels == label] = 0

        logging.info("Process %d: 3D post processing finished", PID)

        np.save(_output_path, prediction)
        logging.info("Process %d: processed prediction numpy array saved", PID)

        gif_path = _output_path.parent / "prediction_psp.gif"

        prediction = prediction * _multipier  # To differentiate the colors
        prediction = prediction.astype(np.uint8)

        with imageio.get_writer(gif_path, mode='I') as writer:
            for index in range(prediction.shape[0]):
                writer.append_data(prediction[index])

        logging.info("Process %d: processed prediction gif saved", PID)
