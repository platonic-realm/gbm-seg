"""Anisotropic 3D Swin Transformer building blocks.

Adapted from Liu et al. *Swin Transformer V1/V2* and Hatamizadeh et al.
*Swin-UNETR* (Med. Image Anal. 2023) to handle the project's shallow Z
stacks. Key deviations from the standard:

  - **Patch embedding is `(1, 2, 2)` strided**: the entry conv halves XY
    but leaves Z untouched, mirroring the input shape of the existing
    Unet3D's first encoder stage.
  - **Patch merging deducts Z by 2 per stage**: ``Conv3d(kernel=(3,2,2),
    stride=(1,2,2), padding=0)``. With ``Z=12`` input, the encoder Z
    progression matches Unet3D's pool (kernel=(3,2,2), stride=(1,2,2)):
    12 → 10 → 8 → 6. MONAI's SwinUNETR halves Z at each merge, which
    collapses to <1 within a few stages on Z=12.
  - **Full-Z window per stage**: each stage uses ``window_size=(Z_stage, 7, 7)``
    where ``Z_stage`` is the current Z at that stage. Captures all Z slices
    at every level — Z is too shallow to gain from splitting it.
  - **Cyclic shift in XY only**: shifting Z is meaningless when the window
    already spans the full Z dimension; the shift vector is ``(0, sh, sw)``
    where ``sh = sw = window_size_xy // 2`` for the shifted blocks.
  - **XY padding**: ``window_partition`` zero-pads H, W to be divisible by
    the window dims, and ``window_reverse`` crops back. Standard Swin idiom.

The relative-position-bias table is computed per-stage for the
*stage-specific* window size, indexed by ``(Δz, Δh, Δw)`` triples.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _to_3tuple(x):
    if isinstance(x, (tuple, list)):
        if len(x) != 3:
            raise ValueError(f"Expected length-3, got {len(x)}: {x}")
        return tuple(int(v) for v in x)
    return (int(x), int(x), int(x))


# ---------------------------------------------------------------------------
# MLP block (used inside the transformer)
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """Standard transformer 2-layer MLP with GELU activation + dropout."""

    def __init__(self, in_features: int, hidden_features: int,
                 out_features: int = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# Window partition / reverse with implicit XY padding
# ---------------------------------------------------------------------------

def window_partition(x: Tensor, window_size: Sequence[int]
                     ) -> tuple[Tensor, tuple[int, int, int]]:
    """Partition ``(B, Z, H, W, C)`` into non-overlapping 3D windows.

    The input is zero-padded on H and W (right side) so each dim is divisible
    by the corresponding window size. Returns the windowed tensor of shape
    ``(B * nW, win_z * win_h * win_w, C)`` and the padding amounts so the
    caller can crop back after attention.
    """
    B, Z, H, W, C = x.shape
    win_z, win_h, win_w = window_size

    pad_h = (win_h - H % win_h) % win_h
    pad_w = (win_w - W % win_w) % win_w
    # Z must already match; window_size[0] should equal Z by design.
    if Z != win_z:
        raise ValueError(
            f"Window Z ({win_z}) must equal input Z ({Z}); "
            "use full-Z windows for shallow stacks.")
    if pad_h or pad_w:
        # F.pad takes pad amounts in reverse-axis order; for (B, Z, H, W, C),
        # we want to pad H and W (axes 2 and 3). Pad takes (W_left, W_right,
        # H_left, H_right) when applied to the last two of the trailing axes
        # — but our last axis is C, so we permute briefly.
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, Z, H, W)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.permute(0, 2, 3, 4, 1)  # back to (B, Z, H+pad, W+pad, C)

    Hp, Wp = H + pad_h, W + pad_w
    # Partition: split each spatial dim into windows.
    x = x.view(B, Z // win_z, win_z,
               Hp // win_h, win_h,
               Wp // win_w, win_w, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = x.view(-1, win_z * win_h * win_w, C)
    return windows, (Z, Hp, Wp)


def window_reverse(windows: Tensor, window_size: Sequence[int],
                   padded_shape: tuple[int, int, int],
                   original_hw: tuple[int, int]) -> Tensor:
    """Inverse of :func:`window_partition`.

    Takes the windowed tensor produced by ``window_partition`` and folds it
    back to ``(B, Z, H, W, C)``, cropping off the implicit XY padding.

    ``padded_shape`` and ``original_hw`` come from ``window_partition``.
    """
    win_z, win_h, win_w = window_size
    Z, Hp, Wp = padded_shape
    nW = (Z // win_z) * (Hp // win_h) * (Wp // win_w)
    B = windows.shape[0] // nW
    C = windows.shape[-1]
    x = windows.view(B, Z // win_z, Hp // win_h, Wp // win_w,
                     win_z, win_h, win_w, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, Z, Hp, Wp, C)
    H, W = original_hw
    if Hp != H or Wp != W:
        x = x[:, :, :H, :W, :].contiguous()
    return x


# ---------------------------------------------------------------------------
# Window attention with relative position bias
# ---------------------------------------------------------------------------

class WindowAttention3D(nn.Module):
    """Multi-head self-attention within a 3D window, with relative position bias.

    Input/output shape: ``(B * nW, win_z * win_h * win_w, C)``.
    """

    def __init__(self, dim: int, window_size: Sequence[int], num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table indexed by (Δz, Δh, Δw). Range of each
        # delta is [-(win-1), win-1], so the table has (2*win-1)^3 entries
        # per head.
        win_z, win_h, win_w = self.window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros(
            (2 * win_z - 1) * (2 * win_h - 1) * (2 * win_w - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute the flat index into the bias table for every pair (i, j)
        # of tokens within a window.
        coords_z = torch.arange(win_z)
        coords_h = torch.arange(win_h)
        coords_w = torch.arange(win_w)
        coords = torch.stack(torch.meshgrid(
            coords_z, coords_h, coords_w, indexing='ij'))  # (3, win_z, win_h, win_w)
        coords_flat = coords.flatten(1)  # (3, N)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 3)
        # Shift to start from 0 in each dim.
        relative_coords[:, :, 0] += win_z - 1
        relative_coords[:, :, 1] += win_h - 1
        relative_coords[:, :, 2] += win_w - 1
        # Flatten into a 1D index.
        relative_coords[:, :, 0] *= (2 * win_h - 1) * (2 * win_w - 1)
        relative_coords[:, :, 1] *= (2 * win_w - 1)
        self.register_buffer(
            'relative_position_index',
            relative_coords.sum(-1).long())  # (N, N)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """``x``: ``(B*nW, N, C)`` where ``N = win_z * win_h * win_w``.
        ``mask``: optional ``(nW, N, N)`` additive mask for shifted windows.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, heads, N, N)

        # Add relative position bias.
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, self.num_heads)
        bias = bias.permute(2, 0, 1).contiguous()  # (heads, N, N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Swin Transformer block (norm → window attention → norm → MLP, with shift)
# ---------------------------------------------------------------------------

class SwinTransformerBlock3D(nn.Module):
    """One Swin block: window-MSA + MLP with optional XY cyclic shift.

    Z is not shifted because the Z window already spans the full Z extent
    (see module docstring). For the shifted variant, only H and W are
    rolled by ``window_size[1:] // 2``.
    """

    def __init__(self, dim: int, num_heads: int,
                 window_size: Sequence[int],
                 shift_size: Sequence[int] = (0, 0, 0),
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = tuple(window_size)
        self.shift_size = tuple(shift_size)
        # Shift must be smaller than window in every axis.
        if any(s >= w for s, w in zip(self.shift_size, self.window_size)):
            raise ValueError(
                f"shift_size {self.shift_size} must be < window_size "
                f"{self.window_size} in every dim.")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size=self.window_size,
                                       num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``(B, Z, H, W, C)``."""
        B, Z, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        do_shift = any(s != 0 for s in self.shift_size)
        if do_shift:
            sz, sh, sw = self.shift_size
            x = torch.roll(x, shifts=(-sz, -sh, -sw), dims=(1, 2, 3))

        windows, padded_shape = window_partition(x, self.window_size)
        mask = self._compute_attn_mask(padded_shape, x.device) if do_shift else None

        attn_out = self.attn(windows, mask=mask)
        x = window_reverse(attn_out, self.window_size, padded_shape, (H, W))

        if do_shift:
            sz, sh, sw = self.shift_size
            x = torch.roll(x, shifts=(sz, sh, sw), dims=(1, 2, 3))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _compute_attn_mask(self, padded_shape, device):
        """Build the additive mask that prevents wraps-from-roll voxels
        in the same window from attending to each other.

        For each spatial axis we partition its index range into regions
        based on the shift and window size; voxels in different region
        triples shouldn't attend.
        """
        win_z, win_h, win_w = self.window_size
        sz, sh, sw = self.shift_size
        Z, Hp, Wp = padded_shape
        img_mask = torch.zeros((1, Z, Hp, Wp, 1), device=device)
        # Region IDs along each axis.
        z_slices = self._axis_slices(Z, win_z, sz)
        h_slices = self._axis_slices(Hp, win_h, sh)
        w_slices = self._axis_slices(Wp, win_w, sw)
        cnt = 0
        for zs in z_slices:
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, zs, hs, ws, :] = cnt
                    cnt += 1
        mask_windows, _ = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.squeeze(-1)  # (nW, N)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    @staticmethod
    def _axis_slices(size: int, win: int, shift: int):
        """Return the index-slice regions for one axis after shift."""
        if shift == 0:
            return [slice(0, size)]
        return [slice(0, size - win),
                slice(size - win, size - shift),
                slice(size - shift, size)]


# ---------------------------------------------------------------------------
# Patch embedding and merging
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """Conv-based patch embedding with anisotropic ``(1, 2, 2)`` stride.

    Input ``(B, C_in, Z, H, W)`` → output ``(B, embed_dim, Z, H/2, W/2)``.
    Followed by layer norm in the feature dim.
    """

    def __init__(self, in_channels: int, embed_dim: int,
                 patch_size: Sequence[int] = (1, 2, 2)):
        super().__init__()
        self.patch_size = _to_3tuple(patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)                       # (B, C, Z, H', W')
        x = x.permute(0, 2, 3, 4, 1)            # (B, Z, H', W', C)
        return self.norm(x)


class PatchMerging3D(nn.Module):
    """``Conv3d(kernel=(3,2,2), stride=(1,2,2))`` patch merging.

    Z deducts by 2 per call, XY halves, channels double. Replaces Swin's
    classic concat-4-corners-then-linear pattern (which keeps Z) with a
    conv-based merge that matches Unet3D's encoder pooling.

    Input/output operates on ``(B, Z, H, W, C)`` to stay consistent with
    the rest of the Swin building blocks; the Conv3d is applied after
    a permute to ``(B, C, Z, H, W)`` and back.
    """

    def __init__(self, dim: int, z_deduction: int = 2):
        super().__init__()
        z_kernel = z_deduction + 1
        self.z_kernel = z_kernel
        self.proj = nn.Conv3d(dim, 2 * dim,
                              kernel_size=(z_kernel, 2, 2),
                              stride=(1, 2, 2),
                              padding=0)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``(B, Z, H, W, C)`` with even H and W and ``Z >= z_kernel``."""
        B, Z, H, W, C = x.shape
        if Z < self.z_kernel:
            raise ValueError(
                f"PatchMerging3D got Z={Z} but needs Z >= {self.z_kernel}.")
        # Pad H/W to even so the (1,2,2) stride is well-defined.
        pad_h = H % 2
        pad_w = W % 2
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, Z, H, W)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)                # (B, 2C, Z-z_ded, H/2, W/2)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # back to (B, ..., 2C)
        return self.norm(x)


# ---------------------------------------------------------------------------
# A "stage" of Swin: a sequence of paired regular+shifted transformer blocks.
# ---------------------------------------------------------------------------

class BasicLayer3D(nn.Module):
    """A stage of Swin blocks: ``depth`` paired regular/shifted blocks at
    a fixed spatial resolution, optionally followed by a ``PatchMerging3D``
    downsample.
    """

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: Sequence[int],
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 downsample: bool = False):
        super().__init__()
        win_z, win_h, win_w = window_size
        # Z is full-window, so no Z shift. XY shifts by half the window.
        shift = (0, win_h // 2, win_w // 2)

        blocks = []
        for i in range(depth):
            blocks.append(SwinTransformerBlock3D(
                dim=dim, num_heads=num_heads,
                window_size=window_size,
                shift_size=shift if (i % 2 == 1) else (0, 0, 0),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop))
        self.blocks = nn.ModuleList(blocks)
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(features_after_blocks, features_after_optional_downsample)``.

        Skip connections take ``features_after_blocks`` (before downsampling).
        """
        for block in self.blocks:
            x = block(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return skip, x
