# Python Imports
import os

import imageio

# Library Imports
import numpy as np
import tifffile
import torch
from numpy import array
from scipy import ndimage
from skimage import measure, morphology
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local Imports
from src.infer.stitching import StitchAccumulator
from src.train.snapper import Snapper
from src.utils.misc import create_dirs_recursively


class Inference:
    """Slides an inference DataLoader through the model and stitches patches.

    A3 (Phase 6): the full-volume accumulator lives here, not in the model.
    The stitching mode is config-driven via ``inference.stitching``
    (gaussian / hann / sum_logits / flat_softmax). See
    ``src/infer/stitching.py`` for the four-mode contract.
    """

    def __init__(self,
                 _model: nn.Module,
                 _data_loader: DataLoader,
                 _snapper: Snapper,
                 _device: str,
                 _results_path: str,
                 _snapshot_path: str,
                 _dp: bool,
                 _interpolate: bool,
                 _scale_factor: int,
                 _stitching_mode: str,
                 _patch_size,
                 _result_shape):

        self.data_loader = _data_loader
        self.device = _device
        self.results_path = _results_path

        self.interpolate = _interpolate
        self.scale_factor = _scale_factor

        _model.to(self.device)
        # Inference path: only the model weights need restoring; no
        # optimiser/scheduler/stepper/RNG. Pass the explicit snapshot
        # path so the auto-discovery (looks in <path>/continue/) is
        # bypassed.
        _snapper.load(_model, _device=self.device, _path=_snapshot_path,
                      _restore_rng=False)

        self.model = _model
        self.model.eval()

        self.accumulator = StitchAccumulator(
            mode=_stitching_mode,
            patch_size=_patch_size,
            result_shape=_result_shape)

    def infer(self) -> None:
        """Run the full inference loop and write predictions to ``results_path``."""
        for data in tqdm(self.data_loader,
                         desc="Segmentation"):

            sample = data['sample'].to(self.device)
            offsets = data['offsets'].to(self.device)

            with torch.no_grad():
                logits, _ = self.model(sample)

            # C1.2: when the model was trained with deep supervision, it
            # returns a list ``[final_logits, aux_logits, ...]``. Inference
            # uses only the final-resolution head.
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            self.accumulator.add_batch(logits, offsets)

        output_dir = os.path.join(self.results_path,
                                  self.data_loader.dataset.file_name)

        create_dirs_recursively(os.path.join(output_dir, "dummy"))

        result = self.accumulator.finalize()

        del offsets
        del sample

        self.save_result(self.data_loader.dataset.nephrin,
                         self.data_loader.dataset.collagen4,
                         self.data_loader.dataset.wga,
                         result,
                         output_dir,
                         self.data_loader.dataset.tiff_tags)

    def blender_visualization(self,
                              _distance_results: array,
                              _fd_results: array,
                              _output_path: str):

        create_dirs_recursively(os.path.join(_output_path, "dummy"))

        mean = np.mean(_distance_results[_distance_results != 0])
        first_layer = _distance_results[0]
        first_layer[first_layer != 0] = mean/2
        _distance_results[0] = first_layer

        last_layer = _distance_results[_distance_results.shape[0]-1]
        last_layer[last_layer != 0] = mean/2
        _distance_results[_distance_results.shape[0]-1] = last_layer

        pad_width = ((1, 1), (1, 1), (1, 1))

        _distance_results = np.pad(_distance_results, pad_width, mode="constant", constant_values=0)
        verts, faces, normals, values = measure.marching_cubes(volume=_distance_results,
                                                               level=0.1,
                                                               step_size=1.1,
                                                               allow_degenerate=False)

        np.savez_compressed(os.path.join(_output_path, "verts_distance.npz"), arr=verts)
        np.savez_compressed(os.path.join(_output_path, "faces_distance.npz"), arr=faces)
        np.savez_compressed(os.path.join(_output_path, "values_distance.npz"), arr=values)

        mean = np.mean(_fd_results[_fd_results != 0])
        first_layer = _fd_results[0]
        first_layer[first_layer != 0] = mean/2
        _fd_results[0] = first_layer

        last_layer = _fd_results[_fd_results.shape[0]-1]
        last_layer[last_layer != 0] = mean/2
        _fd_results[_fd_results.shape[0]-1] = last_layer

        _fd_results = np.pad(_fd_results, pad_width, mode="constant", constant_values=0)
        verts, faces, normals, values = measure.marching_cubes(volume=_fd_results,
                                                               level=0.1,
                                                               step_size=1.1,
                                                               allow_degenerate=False)

        np.savez_compressed(os.path.join(_output_path, "verts_bumpiness.npz"), arr=verts)
        np.savez_compressed(os.path.join(_output_path, "faces_bumpiness.npz"), arr=faces)
        np.savez_compressed(os.path.join(_output_path, "values_bumpiness.npz"), arr=values)

    def save_result(self,
                    _nephrin: array,
                    _wga: array,
                    _collagen4: array,
                    _prediction: array,
                    _output_path: str,
                    _tiff_tags: dict,
                    _multiplier: int = 120):

        np.savez_compressed(os.path.join(_output_path, "prediction.npz"),
                arr=_prediction)

        prediction_tif_path = os.path.join(_output_path, "prediction.tif")
        prediction_gif_path = os.path.join(_output_path, "prediction.gif")

        _prediction = _prediction * _multiplier
        _prediction = _prediction.numpy().astype(np.uint8)

        with imageio.get_writer(prediction_gif_path, mode='I') as writer:
            for index in range(_prediction.shape[0]):
                writer.append_data(_prediction[index])

        _prediction = np.stack([_nephrin,
                                _wga,
                                _collagen4,
                                _prediction],
                               axis=1)

        tifffile.imwrite(prediction_tif_path,
                         _prediction,
                         shape=_prediction.shape,
                         imagej=True,
                         metadata={'axes': 'ZCYX', 'fps': 10.0},
                         compression='lzw')

    def post_processing(self,
                        _prediction: Tensor):
        prediction = _prediction.detach().cpu().numpy()
        if not self.psp_enabled:
            return prediction

        for i in range(prediction.shape[0]):

            kernel = morphology.rectangle(self.psp_kernel_size,
                                          self.psp_kernel_size)
            eroded_image = morphology.erosion(prediction[i, :, :],
                                              kernel)
            prediction[i, :, :] = eroded_image

            sample = prediction[i, :, :]
            labels = measure.label(sample,
                                   connectivity=1)
            # count pixels in each connected component
            unique_labels, label_counts = np.unique(labels,
                                                    return_counts=True)
            # remove small connected components
            # print(f"mean: {int(np.mean(label_counts))},\
            #       std: {int(np.std(label_counts))},\
            #       max: {int(np.max(label_counts))},\
            #       min: {int(np.min(label_counts))}")

            for label, count in zip(unique_labels,
                                    label_counts):
                if count < self.psp_obj_min_size and label != 0:
                    prediction[i, :, :][labels == label] = 0

            kernel = morphology.rectangle(self.psp_kernel_size,
                                          self.psp_kernel_size)
            dilated_image = morphology.dilation(prediction[i, :, :],
                                                kernel)
            prediction[i, :, :] = dilated_image

        labels, labels_num = ndimage.label(prediction)
        # count pixels in each connected component
        for label_index in range(labels_num):
            voxel_count = np.sum(labels == label_index)
            if voxel_count < 1000:
                prediction[labels == label] = 0

        return prediction
