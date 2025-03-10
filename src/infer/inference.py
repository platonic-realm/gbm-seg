# Python Imports
import os

# Library Imports
import numpy as np
from numpy import array
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import tifffile
import imageio
from tqdm import tqdm
from skimage import measure, morphology
from scipy import ndimage

# Local Imports
from src.train.snapper import Snapper
from src.infer.morph import Morph

from src.utils.misc import create_dirs_recursively


class Inference():
    def __init__(self,
                 _model: nn.Module,
                 _data_loader: DataLoader,
                 _snapper: Snapper,
                 _device: str,
                 _results_path: str,
                 _snapshot_path: str,
                 _dp: bool,
                 _post_processing: bool,
                 _psp_obj_min_size: int,
                 _psp_kernel_size: int,
                 _interpolate: bool,
                 _scale_factor: int):

        self.data_loader = _data_loader
        self.device = _device
        self.results_path = _results_path
        self.psp_enabled = _post_processing
        self.psp_obj_min_size = _psp_obj_min_size
        self.psp_kernel_size = _psp_kernel_size

        self.interpolate = _interpolate
        self.scale_factor = _scale_factor

        _model.to(self.device)
        _snapper.load(_model, self.device, _snapshot_path)

        self.model = _model
        self.model.eval()

    def infer(self):

        for data in tqdm(self.data_loader,
                         desc="Segmentation"):

            sample = data['sample'].to(self.device)
            offsets = data['offsets'].to(self.device)

            with torch.no_grad():
                _, _ = self.model(sample, offsets)

        output_dir = os.path.join(self.results_path,
                                  self.data_loader.dataset.file_name)

        create_dirs_recursively(os.path.join(output_dir, "dummy"))

        result = self.model.module.get_result()
        result = torch.argmax(result, dim=0)
        result = self.post_processing(result)

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

        np.save(os.path.join(_output_path, "verts_distance.npy"), verts)
        np.save(os.path.join(_output_path, "faces_distance.npy"), faces)
        np.save(os.path.join(_output_path, "values_distance.npy"), values)

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

        np.save(os.path.join(_output_path, "verts_bumpiness.npy"), verts)
        np.save(os.path.join(_output_path, "faces_bumpiness.npy"), faces)
        np.save(os.path.join(_output_path, "values_bumpiness.npy"), values)

    def save_result(self,
                    _nephrin: array,
                    _wga: array,
                    _collagen4: array,
                    _prediction: array,
                    _output_path: str,
                    _tiff_tags: dict,
                    _multiplier: int = 120):

        prediction_tif_path = os.path.join(_output_path, "prediction.tif")
        prediction_gif_path = os.path.join(_output_path, "prediction.gif")

        _prediction = _prediction * _multiplier
        _prediction = _prediction.astype(np.uint8)

        np.save(os.path.join(_output_path,
                             "prediction.npy"),
                _prediction)

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

            kernel = morphology.rectangle(self.psp_kernel_size, self.psp_kernel_size)
            eroded_image = morphology.erosion(prediction[i, :, :], kernel)
            prediction[i, :, :] = eroded_image

            sample = prediction[i, :, :]
            labels = measure.label(sample, connectivity=1)
            # count pixels in each connected component
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            # remove small connected components
            # print(f"mean: {int(np.mean(label_counts))}, std: {int(np.std(label_counts))}, max: {int(np.max(label_counts))}, min: {int(np.min(label_counts))}")
            for label, count in zip(unique_labels, label_counts):
                if count < self.psp_obj_min_size and label != 0:
                    prediction[i, :, :][labels == label] = 0

            kernel = morphology.rectangle(self.psp_kernel_size, self.psp_kernel_size)
            dilated_image = morphology.dilation(prediction[i, :, :], kernel)
            prediction[i, :, :] = dilated_image

        labels, labels_num = ndimage.label(prediction)
        # count pixels in each connected component
        for label_index in range(labels_num):
            voxel_count = np.sum(labels == label_index)
            if voxel_count < 1000:
                prediction[labels == label] = 0

        return prediction
