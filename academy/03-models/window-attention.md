# Window attention and relative position bias

This doc unpacks the attention mechanism inside [SwinUNETR](swinunetr.md).
If you're comfortable with vanilla transformer attention but not Swin
specifics, this is the missing bridge.

## Why "window" attention

Full self-attention on a `(B, Z, H, W, C)` feature map has cost
quadratic in `Z*H*W` (the number of tokens). For a Z=12, 256×256
input at stage 0, that's `12*256*256 = 786,432` tokens — completely
infeasible to do full attention over.

**Swin's solution**: partition the volume into non-overlapping spatial
windows (`window_z × window_h × window_w` voxels each), compute
attention **only within each window**, then alternately *shift* the
windows by half a window so information can flow across window
boundaries over multiple layers.

In our 3D version, the window is `(stage_z, 7, 7)` by default (see
[swinunetr](swinunetr.md)). At stage 0 with Z=12, that's
`12 × 7 × 7 = 588 tokens per window`. Attention cost is now quadratic
in that, not in the full image — manageable.

## The relative position bias

Plain attention is permutation-equivariant: `attention(Q, K, V)` is
unchanged if you permute the rows of K and V together. Swin (and the
original transformer) breaks this with **position information**.
Vision Transformers usually add an *absolute* position embedding to
the input tokens; Swin instead adds a **relative position bias** to
the attention scores:

```
attention(Q, K, V) = softmax( (Q K^T / sqrt(d)) + B ) V
```

where `B[i, j]` depends only on the relative offset between tokens `i`
and `j` within the window, **not** on their absolute positions. This
gives the model a way to learn "tokens 3 apart in Z attend to each
other differently from tokens 6 apart" — but the same bias applies
regardless of where the window sits in the overall volume.

## How B is parameterised

For each attention head, we maintain a small table of size
`(2*win_z - 1) × (2*win_h - 1) × (2*win_w - 1)`, flattened into 1D.
The bias for tokens at offset `(Δz, Δh, Δw)` is the table entry at
index `(Δz + win_z - 1, Δh + win_h - 1, Δw + win_w - 1)` (shift to
non-negative, then ravel).

`src/models/swin_unetr/blocks.py:154-177`:

```python
self.relative_position_bias_table = nn.Parameter(torch.zeros(
    (2 * win_z - 1) * (2 * win_h - 1) * (2 * win_w - 1), num_heads))
nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
```

This table is **learned** (a `nn.Parameter`) — gradient descent picks
the values that minimise the loss. The `relative_position_index`
buffer precomputes the table lookup for every `(i, j)` token pair so
the forward pass is just an indexed gather.

For our defaults at stage 0 (`(win_z, win_h, win_w) = (12, 7, 7)`,
`num_heads=3`): the table has `23 × 13 × 13 = 3,887` entries × 3 heads
= **11,661 learnable parameters** at this stage alone. Across all four
stages and many attention layers, the total is in the tens of
thousands of parameters — meaningful but not dominant.

## Where the bias is added in the forward pass

`src/models/swin_unetr/blocks.py:186-232` uses
`F.scaled_dot_product_attention` (SDPA — a fused, memory-efficient
implementation in PyTorch ≥ 2.0). The bias is passed as the
`attn_mask` argument:

```python
bias = self.relative_position_bias_table[
    self.relative_position_index.view(-1)
].view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
# bias shape: (heads, N, N)

if mask is None:
    out = F.scaled_dot_product_attention(q, k, v,
            attn_mask=bias.unsqueeze(0),
            dropout_p=dropout_p, scale=self.scale)
```

The shifted-window mask (when the block is the "shifted" half of a
regular/shifted pair) is *added* to the bias before being passed in —
they're both additive on the pre-softmax scores. Tests
`test_window_attention_sdpa_matches_reference_no_mask` and
`...with_mask` verify equivalence to a reference plain-softmax
implementation.

## SDPA — why bother

Without SDPA, computing attention requires materialising the full
`(B, heads, N, N)` score matrix in memory before softmax. For our
windows with `N = win_z * win_h * win_w = 588` at stage 0 and batch
sizes of 4-8, that tensor is **billions of entries** — gigabytes per
forward. SDPA fuses softmax + matmul so this is never written to
memory.

This is the difference between "Swin trains on our hardware" and "OOM
on every step". Even with SDPA, the per-rank batch on 40 GB A100s is
capped at 4 for `[12, 256, 256]` patches — see [the memory wall
section in the swin model doc](swinunetr.md#memory-wall).

## Killing the bias — the v4 ablation

Commit `03bacc5` added `use_relative_pos_bias: bool` as a config knob.
When False:

- `WindowAttention3D.__init__` skips the table allocation and the
  precomputed index, registering them as `None`.
- `WindowAttention3D.forward` passes `attn_mask=mask` (or `None` if
  unshifted) instead of `bias + mask`. The attention still runs at
  the same cost; it just loses the per-(Δz, Δh, Δw) scalar that
  encodes position-specific patterns.

What this targets: the [Z-jaggedness
problem](../07-case-study-z-jaggedness/README.md). The bias is the
mechanism by which the Swin encoder can learn "label transitions live
at Δz = ±6 within the window" — the period that matches the stacked-label
artifact in the upsampled training target. Removing the bias **forces
the attention to be translation-equivariant in Z**, in the same way the
conv U-Net always is.

The downside: the bias also helps with **legitimate** position-dependent
features (the GBM is near the centre of the patch more often than
near the edges; this is a real signal). Disabling it broadly may hurt
Dice. That's exactly the question the v4 ablation answers.

## What about absolute position embeddings?

Swin's original design uses *only* the relative position bias; no
absolute position embedding. So when `use_relative_pos_bias=False`,
the attention has **no spatial-prior information at all** — it's
purely set-attention. The only spatial structure left in the model
comes from:
- The `PatchEmbed3D` conv (with a fixed kernel layout).
- The `PatchMerging3D` convs in the encoder.
- The `_PatchExpand3D` and `_ConvFusion3D` convs in the decoder.

All four use 3D convolutions, which are translation-equivariant. So
**disabling the bias makes the entire model translation-equivariant
in the same way the conv U-Net is**. That's the heart of the v4_nobias
hypothesis: maybe what makes U-Net's Z output smooth is exactly that
property, and giving it to SwinUNETR is the simplest fix.

## Tests

`tests/models/test_swin_unetr.py` covers:

- `test_window_attention_sdpa_matches_reference_no_mask` — SDPA == plain
  softmax attention (no mask).
- `test_window_attention_sdpa_matches_reference_with_mask` — same with
  shifted-window mask.
- `test_window_attention_no_bias_skips_table` — flag-off path drops
  the parameter and the index buffer.
- `test_window_attention_no_bias_forward_runs` — both no-mask and
  masked paths still produce correct shapes with no NaN.
- `test_swin_unetr_no_bias_has_fewer_params` — sanity that the
  ablation actually removes parameters from the model.
