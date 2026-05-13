# Blender preparation, rendering, and export.
# Moved out of src/utils/misc.py during the Phase 3 split.

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile
import torch
from numpy import array
from skimage import measure
from skimage.transform import resize

from src.utils.misc import create_dirs_recursively


def blender_visualization(_distance_results: array, _output_path: str):
    create_dirs_recursively(os.path.join(_output_path, "dummy"))

    mean = np.mean(_distance_results[_distance_results != 0])
    first_layer = _distance_results[0]
    first_layer[first_layer != 0] = mean / 2
    _distance_results[0] = first_layer

    last_layer = _distance_results[_distance_results.shape[0] - 1]
    last_layer[last_layer != 0] = mean / 2
    _distance_results[_distance_results.shape[0] - 1] = last_layer

    pad_width = ((1, 1), (1, 1), (1, 1))
    _distance_results = np.pad(_distance_results, pad_width, mode="constant",
                               constant_values=0)
    verts, faces, normals, values = measure.marching_cubes(
        volume=_distance_results, level=0.1, allow_degenerate=False)

    np.savez_compressed(os.path.join(_output_path, "verts_distance.npz"), arr=verts)
    np.savez_compressed(os.path.join(_output_path, "faces_distance.npz"), arr=faces)
    np.savez_compressed(os.path.join(_output_path, "values_distance.npz"), arr=values)


def blender_prepare(_sample_dir: str) -> None:
    sample = Path(_sample_dir)
    distance_result = np.load(sample / "distance_result.npz")['arr']
    blender_visualization(_distance_results=distance_result,
                          _output_path=sample / "blender/")


def blender_render(_inference_dir: str) -> None:
    num_gpus = torch.cuda.device_count()
    gpu_index = 0
    gpu_capacity = 5

    all_files = os.listdir(_inference_dir)
    directories = [item for item in all_files
                   if os.path.isdir(os.path.join(_inference_dir, item))
                   and item not in ['.', '..']]
    commands = []
    for directory in directories:
        verts_path = os.path.join(_inference_dir, directory, "blender", "verts_distance.npz")
        faces_path = os.path.join(_inference_dir, directory, "blender", "faces_distance.npz")
        values_path = os.path.join(_inference_dir, directory, "blender", "values_distance.npz")
        result_blend_path = os.path.join(_inference_dir, directory, "blender", "result_distance.blend")
        result_anim_path = os.path.join(_inference_dir, directory, "blender", "result_distance.mp4")
        commands.append(
            f"export CUDA_VISIBLE_DEVICES={gpu_index}; "
            f"blender --background --python src/scripts/blender_render.py -- "
            f"res/blender_template.blend "
            f"{shlex.quote(verts_path)} {shlex.quote(faces_path)} {shlex.quote(values_path)} "
            f"{shlex.quote(result_blend_path)} {shlex.quote(result_anim_path)}")
        gpu_index = (gpu_index + 1) % num_gpus

    compute_limit = num_gpus * gpu_capacity
    for i in range(int(len(commands) / compute_limit) + 1):
        sub_commands = commands[i * compute_limit:(i + 1) * compute_limit]
        processes = [subprocess.Popen(cmd, shell=True, executable='/bin/bash')
                     for cmd in sub_commands]
        for proc in processes:
            proc.wait()


def export_results(_inference_result_path: Path, _inference_export_path: Path):
    scale_factor = 1 / 3

    for dir in _inference_result_path.iterdir():
        if not dir.is_dir():
            continue

        name = dir.name

        tiff_file = _inference_export_path / "tiff" / name
        thickness_blend = _inference_export_path / "blend" / f"{name} _thickness.blend"
        thickness_mp4 = _inference_export_path / "mp4" / f"{name}_thickness.mp4"
        bumpiness_blend = _inference_export_path / "blend" / f"{name}_bumpiness.blend"
        bumpiness_mp4 = _inference_export_path / "mp4" / f"{name}_bumpiness.mp4"

        create_dirs_recursively(tiff_file)
        create_dirs_recursively(thickness_mp4)
        create_dirs_recursively(thickness_blend)

        shutil.copy(dir / "blender/result_distance.blend", thickness_blend)
        shutil.copy(dir / "blender/result_distance.mp4", thickness_mp4)

        shutil.copy(dir / "blender/result_bumpiness.blend", bumpiness_blend)
        shutil.copy(dir / "blender/result_bumpiness.mp4", bumpiness_mp4)

        labels = np.load(dir / "prediction_psp.npz")['arr']
        labels[labels != 0] = 128

        with tifffile.TiffFile(dir / "prediction.tif") as tif:
            data = tif.asarray()

        data[:, 3, :, :] = labels

        if data.ndim != 4:
            raise ValueError("Expected a 4D TIFF file with shape (channels, z, y, x)")

        z, channels, y, x = data.shape
        new_y, new_x = int(y * scale_factor), int(x * scale_factor)

        downsampled = np.empty((z, channels, new_y, new_x), dtype=data.dtype)
        for i in range(z):
            for c in range(channels):
                downsampled[i, c] = resize(
                    data[i, c], (new_y, new_x),
                    preserve_range=True, anti_aliasing=True).astype(data.dtype)

        tifffile.imwrite(tiff_file, downsampled,
                         shape=downsampled.shape,
                         imagej=True,
                         metadata={'axes': 'ZCYX', 'fps': 10.0},
                         compression='lzw')
