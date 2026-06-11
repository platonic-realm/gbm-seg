"""Treat the ground-truth labels of the *training* set as if they were
model predictions, so the downstream pipeline (psp, morph, blender, render,
export, stats) can be applied to the labels themselves. This gives a
"perfect-model" baseline for every morphometric / statistical figure the
pipeline produces — useful for separating "the model is wrong" from "the
morph algorithm or stats pipeline is wrong".

Source data
-----------
``<exp>/datasets/ds_train/*.tif`` — the per-experiment training TIFFs that
``gbm.py create`` populated. Each TIFF has shape ``(Z, C, H, W)`` with
``C = 4`` (nephrin, collagen-4, WGA, *labels*). These TIFFs are already
at the **upsampled** Z grid (``np.repeat`` applied by create), so the
label channel can be used directly without further upsampling.

The ``--z-repeat N`` flag applies an extra ``np.repeat(..., N, axis=0)``
on top of whatever is in the source. Defaults to 1 (no-op). Use it only
when the source TIFFs are at native Z — for the standard ds_train flow,
leave it at the default.

Output
------
Standard inference-output layout under
``<exp>/results-infer/<output_tag>/<sample_name>/``:

* ``prediction.npz`` — binary mask (Z, H, W), int64 (PSP downstream
  expects int64; matches the model-inference convention).
* ``prediction.tif`` — channels + appended mask (Z, C+1, H, W), float32.

After this script, ``gbm.py psp / morph / stats`` can be run against the
tag exactly as for a model-inference output.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile


def _load_z_scale(_root_path: str, _name: str) -> int:
    """Read the experiment-local z_scale (written by gbm.py create).
    Default to 6 to match the project default if absent."""
    import yaml
    cfg_path = os.path.join(_root_path, _name, 'configs.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Experiment configs missing: {cfg_path}")
    with open(cfg_path, encoding='UTF-8') as f:
        cfg = yaml.safe_load(f) or {}
    return int(cfg.get('trainer', {}).get('data', {}).get('z_scale', 6))


def _list_train_tiffs(_exp_root: str) -> list:
    """Return all training TIFFs under ``<exp>/datasets/ds_train/``."""
    base = Path(_exp_root) / 'datasets' / 'ds_train'
    if not base.is_dir():
        raise FileNotFoundError(
            f"Training-data directory not found: {base}. Expected the "
            "per-experiment ds_train/ populated by `gbm.py create`.")
    paths = sorted(list(base.glob('*.tif')) + list(base.glob('*.tiff')))
    if not paths:
        raise FileNotFoundError(f"No TIFFs found under {base}")
    return paths


def _label_channel_to_mask(_img: np.ndarray) -> np.ndarray:
    """Extract the binary mask channel (last channel) from a (Z, C, H, W)
    labeled TIFF. Returns a (Z, H, W) ``np.uint8`` array of {0, 1}.
    Tolerates the "0/255" convention (uint8 ImageJ output) and the
    "0/1" convention (float32 post-resample) by thresholding at 0.5×max.
    """
    if _img.ndim != 4:
        raise ValueError(
            f"Expected a (Z, C, H, W) TIFF; got shape {_img.shape}. "
            "Training TIFFs have a 4th channel for the mask.")
    label_channel = _img[:, -1, :, :]
    threshold = float(label_channel.max()) * 0.5 if label_channel.max() > 0 else 0.5
    return (label_channel > threshold).astype(np.uint8)


def _write_prediction_outputs(_out_dir: Path,
                              _channels: np.ndarray,
                              _mask: np.ndarray) -> None:
    """Match the inference output layout: prediction.npz (binary mask) +
    prediction.tif (channels + mask appended along C)."""
    _out_dir.mkdir(parents=True, exist_ok=True)
    # prediction.npz: PSP expects int64 (matches src/infer/inferer.py).
    np.savez_compressed(_out_dir / 'prediction.npz', arr=_mask.astype(np.int64))
    # prediction.tif: (Z, C+1, H, W) float32 — mirror the trained-model output.
    stack = np.empty(
        (_channels.shape[0], _channels.shape[1] + 1,
         _channels.shape[2], _channels.shape[3]),
        dtype=np.float32)
    stack[:, :-1, :, :] = _channels.astype(np.float32)
    stack[:, -1, :, :] = _mask.astype(np.float32)
    tifffile.imwrite(_out_dir / 'prediction.tif', stack)


def labels_as_pred(_root_path: str,
                   _name: str,
                   _output_tag: Optional[str] = None,
                   _z_repeat: int = 1) -> str:
    """Materialise the training labels under
    ``<exp>/datasets/ds_train/`` as if they were the model's predictions,
    so the downstream pipeline can be invoked against the resulting tag.

    Parameters
    ----------
    _root_path
        Root experiments path (configs['experiments']['root']).
    _name
        Experiment name.
    _output_tag
        Inference-output tag; defaults to ``'labels_train'``. The
        pipeline-stage CLIs (psp/morph/stats) take ``-it <tag>``.
    _z_repeat
        Extra Z upsampling factor applied via ``np.repeat(..., axis=0)``.
        Defaults to 1 (no extra upsampling) since ds_train/ TIFFs are
        already at the upsampled Z grid produced by ``gbm.py create``.
        Pass 6 only when feeding native-Z labels (rare).

    Returns
    -------
    The output tag (so the caller can chain psp/morph/stats on it).
    """
    output_tag = _output_tag or 'labels_train'
    z_scale = _load_z_scale(_root_path, _name)
    logging.info("labels-as-pred: experiment=%s output_tag=%s "
                 "z_scale=%d z_repeat=%d",
                 _name, output_tag, z_scale, _z_repeat)

    exp_root = os.path.join(_root_path, _name)
    tiffs = _list_train_tiffs(exp_root)
    logging.info("labels-as-pred: found %d training TIFFs in ds_train/",
                 len(tiffs))

    out_root = Path(exp_root) / 'results-infer' / output_tag
    out_root.mkdir(parents=True, exist_ok=True)
    logging.info("labels-as-pred: writing to %s", out_root)

    for tiff_path in tiffs:
        sample_name = tiff_path.name
        sample_out = out_root / sample_name
        logging.info("Processing %s", sample_name)

        img = tifffile.imread(tiff_path)  # (Z, C, H, W)
        if img.ndim != 4 or img.shape[1] < 2:
            logging.warning(
                "Skipping %s: unexpected shape %s (need 4D with ≥2 channels)",
                sample_name, img.shape)
            continue

        channels = img[:, :-1, :, :]
        mask = _label_channel_to_mask(img)
        raw_z = mask.shape[0]

        if _z_repeat > 1:
            mask = np.repeat(mask, _z_repeat, axis=0)
            channels = np.repeat(channels, _z_repeat, axis=0)
            logging.info("  Z=%d → upsampled Z=%d (np.repeat ×%d)",
                         raw_z, mask.shape[0], _z_repeat)
        else:
            logging.info("  Z=%d (no extra upsample; source already at "
                         "upsampled grid)", raw_z)

        _write_prediction_outputs(sample_out, channels, mask)

    logging.info("labels-as-pred: done. Output tag = '%s'", output_tag)
    logging.info(
        "Run: gbm.py psp %s -it %s   (then morph / blender / render / "
        "export / stats — same as a model-inference tag)",
        _name, output_tag)
    return output_tag
