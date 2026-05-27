"""Side-by-side demo: stacked-repeat vs trilinear+threshold label Z-upsampling.

Picks one labeled TIFF from the source dataset (``ds_mouse/ds_train/``),
applies the XY resize used by ``gbm.py create`` to put it on the target
voxel grid, then writes out **two** Z-upsampled variants of the volume:

* ``<stem>_stacked.tiff``     — current behaviour: labels via ``np.repeat``.
* ``<stem>_trilinear.tiff``   — proposed: labels via 3D trilinear interp
                                + threshold at 127.5 (re-binarise to {0,255}).

Open both in ImageJ/Fiji and step through the Z slider on the **label
channel (4th channel)** to see the difference. The stacked variant shows
the GBM label as N identical Z slices in a row (stepped transitions every
``z_scale`` slices); the trilinear variant shows a smoothly-extending
membrane.

Image channels (0..2: nephrin/collagen-4/WGA) are identical between the
two outputs — the only diff is channel 3.

Run:
    python src/scripts/compare_label_upsampling.py [<src_tiff>] \
        [--out-dir /tmp/label_upsample_demo] [--z-scale 6]

If ``<src_tiff>`` is omitted, picks the first 4-channel TIFF found under
``configs.experiments.default_data_path/ds_train/``.
"""
# Python Imports
import argparse
import os
import sys
from pathlib import Path

# Library Imports
import numpy as np
import tifffile
import torch
import torch.nn.functional as Fn
import yaml
from scipy.ndimage import zoom

# Local Imports
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.misc import get_voxel_size                    # noqa: E402


def _xy_resize_to_target(_voxel_space: np.ndarray,
                         _voxel_size,
                         _target_size) -> np.ndarray:
    """Mirror the per-slice XY resize from ``resize_and_copy``.

    Returns a ``(Z, 4, H', W')`` float32 array; the label channel is
    re-binarised to {0, 255} after the zoom, exactly as the production
    code does.
    """
    zoom_factors = (_target_size[0] / _voxel_size[0],
                    _target_size[1] / _voxel_size[1])
    z, c, _, _ = _voxel_space.shape
    has_labels = (c == 4)
    out = None
    for i in range(z):
        slices = [
            zoom(_voxel_space[i, ch], zoom_factors, order=1, prefilter=False)
            for ch in range(3)
        ]
        if has_labels:
            lab = zoom(_voxel_space[i, 3], zoom_factors, order=1, prefilter=False)
            lab = (lab >= 0.5).astype(np.float32) * 255.0
            slices.append(lab)
        if out is None:
            H, W = slices[0].shape
            out = np.zeros((z, c, H, W), dtype=np.float32)
        for ch in range(c):
            out[i, ch] = slices[ch].astype(np.float32)
    return out


def _z_upsample_stacked(_image: np.ndarray, _scale_factor: int) -> np.ndarray:
    """Current behaviour: channels trilinear, labels np.repeat."""
    z, c, h, w = _image.shape
    out_z = z * _scale_factor
    out = np.empty((out_z, c, h, w), dtype=_image.dtype)
    for ch in range(3):
        tensor = torch.from_numpy(_image[:, ch]).float().unsqueeze(0).unsqueeze(0)
        scaled = Fn.interpolate(tensor,
                                scale_factor=(_scale_factor, 1, 1),
                                mode='trilinear',
                                align_corners=False)
        out[:, ch] = scaled.squeeze(0).squeeze(0).numpy().astype(_image.dtype)
    if c == 4:
        out[:, 3] = np.repeat(_image[:, 3], _scale_factor, axis=0)
    return out


def _z_upsample_trilinear(_image: np.ndarray, _scale_factor: int) -> np.ndarray:
    """Proposed behaviour: labels also trilinear-interpolated then thresholded
    at 127.5 (= half of 255) to re-binarise."""
    z, c, h, w = _image.shape
    out_z = z * _scale_factor
    out = np.empty((out_z, c, h, w), dtype=_image.dtype)
    for ch in range(c):
        tensor = torch.from_numpy(_image[:, ch]).float().unsqueeze(0).unsqueeze(0)
        scaled = Fn.interpolate(tensor,
                                scale_factor=(_scale_factor, 1, 1),
                                mode='trilinear',
                                align_corners=False)
        arr = scaled.squeeze(0).squeeze(0).numpy()
        if c == 4 and ch == 3:
            arr = (arr >= 127.5).astype(_image.dtype) * 255.0
        out[:, ch] = arr.astype(_image.dtype)
    return out


def _pick_default_src() -> Path:
    """Find the first 4-channel TIFF in the configured source dataset."""
    cfg = yaml.safe_load(open(REPO / "configs" / "template.yaml"))
    src_root = Path(cfg['experiments']['default_data_path']) / "ds_train"
    for p in sorted(src_root.iterdir()):
        if p.suffix.lower() in ('.tif', '.tiff'):
            with tifffile.TiffFile(p) as t:
                arr = t.asarray()
            if arr.ndim == 4 and arr.shape[1] == 4:
                return p
    raise SystemExit(f"No 4-channel TIFF found under {src_root}")


def _label_voxel_count(arr: np.ndarray) -> int:
    """How many label voxels are foreground (label >= 127.5)."""
    return int((arr[:, 3] >= 127.5).sum())


def _z_step_count(arr: np.ndarray) -> int:
    """Count Z transitions in the label channel.

    Two Z-adjacent slices differ if their label masks differ in at least
    one voxel. Identical slices = no transition (the stacked-repeat case
    produces N-1 identical Z pairs per source slice).
    """
    lab = (arr[:, 3] >= 127.5)
    return int(np.sum(np.any(lab[1:] != lab[:-1], axis=(1, 2))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", nargs="?", default=None,
                        help="path to a 4-channel labelled TIFF (default: "
                             "first found in ds_mouse/ds_train/)")
    parser.add_argument("--out-dir", default="/tmp/label_upsample_demo",
                        help="where to write the two output TIFFs")
    parser.add_argument("--z-scale", type=int, default=None,
                        help="Z upsample factor (default: "
                             "configs.experiments.default_z_scale)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(REPO / "configs" / "template.yaml"))
    target_size = cfg['experiments']['default_voxel_size']
    if args.z_scale is None:
        args.z_scale = int(cfg['experiments']['default_z_scale'])

    src = Path(args.src) if args.src else _pick_default_src()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source TIFF        : {src}")
    print(f"target voxel_size  : {target_size}  (XY zoom)")
    print(f"z_scale            : {args.z_scale}  (Z upsample factor)")
    print(f"Out dir            : {out_dir}")
    print()

    with tifffile.TiffFile(src) as tiff:
        voxel_space = tiff.asarray().astype(np.float32)
        voxel_size = get_voxel_size(tiff, _path=src, _default=target_size)
        metadata = tiff.imagej_metadata or tiff.metadata

    if voxel_space.ndim != 4 or voxel_space.shape[1] != 4:
        raise SystemExit(
            f"Need a 4-channel ZCYX TIFF; got shape {voxel_space.shape}")

    print(f"raw shape          : {voxel_space.shape}  (Z, C, H, W)")
    print("Running XY resize ...")
    xy_resized = _xy_resize_to_target(voxel_space, voxel_size, target_size)
    print(f"after XY resize    : {xy_resized.shape}")
    print()

    print("Running Z upsample — stacked (current)  ...")
    stacked = _z_upsample_stacked(xy_resized, args.z_scale)
    print("Running Z upsample — trilinear (proposed) ...")
    trilinear = _z_upsample_trilinear(xy_resized, args.z_scale)

    # Update ImageJ Z spacing so Fiji shows correct axis units.
    if isinstance(metadata, dict) and metadata.get('spacing'):
        try:
            metadata['spacing'] = float(metadata['spacing']) / args.z_scale
        except (TypeError, ValueError):
            pass

    stem = src.stem
    stacked_path = out_dir / f"{stem}_stacked.tiff"
    trilinear_path = out_dir / f"{stem}_trilinear.tiff"

    # bigtiff=True is required because a Z-upsampled volume at the
    # post-XY-resize voxel grid overflows the 4GB classic-TIFF offsets.
    # imagej=True is incompatible with bigtiff, so drop the ImageJ flag —
    # Fiji opens BigTIFFs fine; the ZCYX axis order is auto-detected from
    # the metadata header below.
    # bigtiff requires JSON-serialisable metadata, but tifffile's imagej_metadata
    # can carry numpy arrays — drop unserialisable entries.
    src_meta = dict(metadata) if isinstance(metadata, dict) else {}
    clean_meta = {'axes': 'ZCYX'}
    for key, val in src_meta.items():
        if isinstance(val, (str, int, float, bool, type(None))):
            clean_meta[key] = val

    tifffile.imwrite(stacked_path, stacked,
                     shape=stacked.shape,
                     bigtiff=True,
                     metadata=clean_meta, compression='lzw')
    tifffile.imwrite(trilinear_path, trilinear,
                     shape=trilinear.shape,
                     bigtiff=True,
                     metadata=clean_meta, compression='lzw')

    print()
    print("== Aggregate stats ==")
    print(f"  {'metric':<28} {'stacked':>14} {'trilinear':>14}")
    print(f"  {'output shape':<28} "
          f"{str(stacked.shape):>14} {str(trilinear.shape):>14}")
    print(f"  {'foreground label voxels':<28} "
          f"{_label_voxel_count(stacked):>14,} "
          f"{_label_voxel_count(trilinear):>14,}")
    print(f"  {'Z-transitions in label':<28} "
          f"{_z_step_count(stacked):>14} "
          f"{_z_step_count(trilinear):>14}")

    # Per-Z foreground count: the actually-revealing comparison. The
    # stacked variant prints `z_scale` identical counts in a row before
    # each transition (the smoking-gun stepped pattern); the trilinear
    # variant prints a smoothly-varying sequence.
    fg_stacked = (stacked[:, 3] >= 127.5).reshape(stacked.shape[0], -1).sum(axis=1)
    fg_tri = (trilinear[:, 3] >= 127.5).reshape(trilinear.shape[0], -1).sum(axis=1)
    print()
    print("== Per-Z foreground voxel count "
          f"(first {min(2 * args.z_scale, len(fg_stacked))} of "
          f"{len(fg_stacked)} slices) ==")
    print(f"  {'Z':>3}  {'stacked':>10}  {'trilinear':>10}")
    for z in range(min(2 * args.z_scale, len(fg_stacked))):
        marker = "  <-- repeat" if (z % args.z_scale != 0) else ""
        print(f"  {z:>3}  {fg_stacked[z]:>10,}  {fg_tri[z]:>10,}{marker}")
    print("  ...")
    print()
    print(f"Written:\n  {stacked_path}\n  {trilinear_path}")
    print()
    print("Open both in Fiji and scroll the Z slider on channel 4 (label).")
    print("The 'stacked' file shows N identical Z slices in a row at every")
    print(f"original-slice boundary (transitions only every {args.z_scale}th Z).")
    print("The 'trilinear' file shows smooth, voxel-grain transitions.")


if __name__ == "__main__":
    sys.exit(main())
