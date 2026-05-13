"""SwinUNETR3D — Swin-Transformer-encoder + U-Net decoder for 3D segmentation.

Anisotropic build for shallow-Z stacks: patch embedding halves XY only,
patch merging deducts Z by 2 (matching Unet3D's encoder pooling), the
window covers the full Z extent at every stage. See ``blocks.py`` for
the per-stage building blocks.

Output contract (post-A3 model interface):
    forward(x) → (logits, outputs)            when deep_supervision is off
    forward(x) → ([logits, aux...], outputs)  when deep_supervision is on

``outputs`` is always ``argmax(softmax(final_logits))`` so metric
computation in the trainer is unaffected by DS.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.models.swin_unetr.blocks import (
    BasicLayer3D,
    PatchEmbed3D,
)


class _ConvFusion3D(nn.Module):
    """Fuse a decoder-up feature with its encoder skip and reduce channels.

    Equivalent to nnU-Net's "decoder block": concat → 2× (Conv3d + Norm + ReLU).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class _PatchExpand3D(nn.Module):
    """Inverse of ``PatchMerging3D``: Z += z_addition, XY *= 2, channels /= 2.

    ``ConvTranspose3d(kernel=(z_addition+1, 2, 2), stride=(1, 2, 2))``.
    """

    def __init__(self, dim: int, out_dim: int = None, z_addition: int = 2):
        super().__init__()
        out_dim = out_dim if out_dim is not None else dim // 2
        self.up = nn.ConvTranspose3d(dim, out_dim,
                                     kernel_size=(z_addition + 1, 2, 2),
                                     stride=(1, 2, 2),
                                     padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, Z, H, W, C) → permute to (B, C, Z, H, W) for ConvTranspose
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.up(x)
        return x.permute(0, 2, 3, 4, 1).contiguous()


class SwinUNETR3D(nn.Module):
    """Swin-UNETR variant for shallow-Z 3D segmentation.

    The model interface mirrors :class:`src.models.unet3d.unet3d.Unet3D` so
    SwinUNETR is drop-in via the model registry: same constructor argument
    names where applicable, same ``forward(x) → (logits, outputs)`` contract
    (or list+argmax when ``_deep_supervision=True``).
    """

    def __init__(
            self,
            _name: str,
            _input_channels: int,
            _number_of_classes: int,
            _sample_dimension: Sequence[int],
            _feature_size: int = 24,
            _depths: Sequence[int] = (2, 2, 2, 2),
            _num_heads: Sequence[int] = (3, 6, 12, 24),
            _window_size_xy: int = 7,
            _mlp_ratio: float = 4.0,
            _qkv_bias: bool = True,
            _drop_rate: float = 0.0,
            _attn_drop_rate: float = 0.0,
            _z_deduction_per_stage: int = 2,
            _deep_supervision: bool = False,
            _ds_levels: int = 2):

        super().__init__()

        if len(_depths) != len(_num_heads):
            raise ValueError("depths and num_heads must have the same length")
        num_stages = len(_depths)
        if num_stages < 2:
            raise ValueError("SwinUNETR3D needs at least 2 encoder stages")

        self.name = _name
        self.sample_dimension = list(_sample_dimension)
        self.input_channels = _input_channels
        self.number_of_classes = _number_of_classes
        self.num_stages = num_stages
        self.z_deduction = int(_z_deduction_per_stage)
        self.window_size_xy = int(_window_size_xy)

        # --- Patch embedding: halve XY, keep Z, project to feature_size ---
        self.patch_embed = PatchEmbed3D(_input_channels, _feature_size,
                                        patch_size=(1, 2, 2))

        # --- Encoder stages ---
        sample_z = self.sample_dimension[0]
        if sample_z - (num_stages - 1) * self.z_deduction <= 0:
            raise ValueError(
                f"sample_dimension Z={sample_z} too shallow for {num_stages} "
                f"stages with z_deduction={self.z_deduction}.")

        encoder_dims = [_feature_size * (2 ** k) for k in range(num_stages)]
        stage_z = [sample_z - k * self.z_deduction for k in range(num_stages)]

        self.encoder_stages = nn.ModuleList()
        for k in range(num_stages):
            window = (stage_z[k], self.window_size_xy, self.window_size_xy)
            self.encoder_stages.append(BasicLayer3D(
                dim=encoder_dims[k],
                depth=_depths[k],
                num_heads=_num_heads[k],
                window_size=window,
                mlp_ratio=_mlp_ratio,
                qkv_bias=_qkv_bias,
                drop=_drop_rate,
                attn_drop=_attn_drop_rate,
                downsample=(k < num_stages - 1)))

        # --- Decoder: mirror of the encoder ---
        # For each non-bottleneck encoder stage k (0..num_stages-2), there is
        # a corresponding decoder up-step that goes from stage (k+1) features
        # back to stage k features, then fuses with skip_k.
        self.decoder_ups = nn.ModuleList()
        self.decoder_fusions = nn.ModuleList()
        for k in reversed(range(num_stages - 1)):
            # k = bottleneck-1, bottleneck-2, ..., 0
            in_dim = encoder_dims[k + 1]
            skip_dim = encoder_dims[k]
            out_dim = encoder_dims[k]
            self.decoder_ups.append(_PatchExpand3D(
                in_dim, out_dim=out_dim, z_addition=self.z_deduction))
            self.decoder_fusions.append(
                _ConvFusion3D(out_dim + skip_dim, out_dim))

        # --- Final XY upsample (mirror of patch embed) + head ---
        self.final_up = nn.ConvTranspose3d(
            encoder_dims[0], encoder_dims[0],
            kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        self.final_head = nn.Conv3d(encoder_dims[0], _number_of_classes,
                                    kernel_size=1)
        self.final_activation = nn.Softmax(dim=1)

        # --- Deep supervision aux heads at decoder mid-levels ---
        self.deep_supervision = _deep_supervision and _ds_levels > 0
        if self.deep_supervision:
            max_ds = len(self.decoder_ups)
            self.ds_levels = min(_ds_levels, max_ds)
            # Aux heads at the OUTPUT of each decoder fusion (deepest first
            # for the first ds_levels). Each head is a 1x1x1 conv from
            # encoder_dims[k] (the fusion output) to num_classes.
            ds_dims = [encoder_dims[num_stages - 2 - i]
                       for i in range(self.ds_levels)]
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(d, _number_of_classes, kernel_size=1)
                for d in ds_dims
            ])
        else:
            self.ds_levels = 0
            self.ds_heads = nn.ModuleList()

    def forward(self, x: Tensor):
        # --- Patch embed ---
        x = self.patch_embed(x)  # (B, Z, H/2, W/2, F)

        # --- Encoder pass, collect skips ---
        skips = []
        for stage in self.encoder_stages:
            skip, x = stage(x)
            skips.append(skip)
        # skips[0] = stage 0 output (shallowest), ..., skips[-1] = bottleneck

        # --- Decoder pass ---
        x = skips[-1]  # bottleneck output (no further downsample for last stage)
        intermediate_features = []  # captured for DS heads (deepest first)
        for up_idx, (up, fuse) in enumerate(zip(self.decoder_ups, self.decoder_fusions)):
            x = up(x)  # ConvTranspose: Z+=z_ded, XY*=2, channels /=2
            # Concatenate with the matching encoder skip (in channel-last NHWDC form
            # for consistency with the Swin path), then permute for the conv fusion.
            skip = skips[self.num_stages - 2 - up_idx]
            x = torch.cat([x, skip], dim=-1)  # (B, Z, H, W, fused_C)
            # Conv fusion expects (B, C, Z, H, W).
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = fuse(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # back to (B, Z, H, W, C)
            if self.deep_supervision and up_idx < self.ds_levels:
                intermediate_features.append(x)

        # --- Final XY upsample + head ---
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, F, Z, H/2, W/2)
        x = self.final_up(x)                          # (B, F, Z, H, W)
        logits = self.final_head(x)                   # (B, num_classes, Z, H, W)
        argmax = torch.argmax(self.final_activation(logits), dim=1)

        if self.deep_supervision:
            # Apply aux heads in deepest-first order, then reverse so the
            # returned list is final → finest_aux → ... → deepest_aux.
            aux_logits = [
                head(feat.permute(0, 4, 1, 2, 3).contiguous())
                for head, feat in zip(self.ds_heads, intermediate_features)
            ]
            aux_logits.reverse()
            return [logits, *aux_logits], argmax

        return logits, argmax
