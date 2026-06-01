# Why two architectures

This repo ships **two trainable segmentation architectures** and treats
both as first-class — neither is a "main" model.

| Name in config | Class | Code |
|---|---|---|
| `unet_3d` | `Unet3D` | `src/models/unet3d/unet3d.py` |
| `swin_unetr` | `SwinUNETR3D` | `src/models/swin_unetr/swin_unetr.py` |

The factory (`src/train/factory.py:createModel`) dispatches on
`configs.trainer.model.name` and builds the right one. Switching
architectures is a one-line config change, nothing else.

## Files in this section

- **[3D U-Net](unet3d.md)** — the convolutional baseline.
- **[Custom SwinUNETR](swinunetr.md)** — the transformer-encoder
  variant. Custom (not MONAI's SwinUNETR) so we control every knob.
- **[Window attention and relative position bias](window-attention.md)**
  — the attention mechanism inside SwinUNETR. Foundational for the
  case study.
- **[Inductive biases — conv vs attention](inductive-biases.md)** —
  the conceptual comparison that explains why they behave differently
  on the same data, especially in Z.

## The bigger picture

Both architectures are evaluated under the same training loop, same
loss, same data pipeline. That's deliberate: it makes the **head-to-head
comparison clean**. When SwinUNETR beats U-Net on expert-set Dice
(0.71 vs 0.64 in the runs as of session-end 2026-05-30), we can say
the architecture is responsible, not some auxiliary detail.

The flipside is that when SwinUNETR's output is visibly *worse* on
some axis (in this project: jagged Z transitions), we can also blame
the architecture. That's the [Z-jaggedness case
study](../07-case-study-z-jaggedness/README.md).

## Shared contract — what both models expose

Every model in this repo presents the same interface so the trainer
and inferer can be model-agnostic:

```python
class Model(nn.Module):
    def forward(self, x: Tensor) -> (logits: Tensor, argmax: Tensor)
```

- `x` has shape `(B, 3, Z, H, W)`. The 3 input channels are nephrin,
  collagen-4, WGA (see [TIFF and channels](../02-data/tiff-and-channels.md)).
- `logits` has shape `(B, num_classes, Z, H, W)`. Pre-softmax.
- `argmax` is the per-voxel class prediction `(B, Z, H, W)` — useful
  for visualisation and for the inferer to assemble the binary mask.

When `deep_supervision=True`, the forward returns
`(list_of_logits, argmax)` where the list is
`[final, finest_aux, …, deepest_aux]`. See
[deep supervision — TODO if you need it now], and the
test in `tests/models/test_swin_unetr.py:test_swin_unetr_ds_returns_list_with_correct_count`.

## Registry

`src/models/__init__.py` keeps a small `MODEL_BUILDERS` dict mapping
the config name to a `build(configs, in_channels, num_classes)`
factory. Adding a new architecture is two steps: write the model class,
add it to the registry. The factory call chain is:

```
gbm.py infer/train
  → src/train/factory.py:createModel
  → src/models/__init__.py:MODEL_BUILDERS[name]
  → src/models/<arch>/__init__.py:build
  → <Arch>3D.__init__(...)
```
