# The Cont loss — weighted CE + Z-continuity term

The `Cont` loss is the headline loss for this project. It's a sum of
two parts: a weighted cross-entropy term and a Z-continuity term.

## Why not just plain cross-entropy

For per-voxel segmentation, the obvious choice is multi-class
cross-entropy with a one-hot target. Two problems on this dataset:

1. **Severe class imbalance.** The GBM occupies a tiny fraction of
   voxels. A model that predicts "all background everywhere" has
   ~95% accuracy and very low CE. We push back with **class weights**
   `[3.0, 7.0]` (background weight 3, GBM weight 7) so getting a GBM
   voxel wrong is ~2.3× more expensive than getting a background
   voxel wrong.

2. **Z-direction artefacts.** As covered in [Z anisotropy](02-data/z-anisotropy-and-upsampling.md),
   the upsampled label has a 6-slice period. Plain CE has no preference
   between "smooth Z predictions" and "matches the stacked target".
   We want to encourage smoothness; the **continuity term** does that
   directly.

## The terms

Implemented in `src/losses/loss_cont.py` (and registered in
`src/losses/__init__.py` so `configs.trainer.optimization.loss.name: Cont`
picks it up). Pseudocode:

```python
def cont_loss(logits, target):
    ce = cross_entropy_with_weights(logits, target, weights=[3.0, 7.0])

    # Continuity term: penalise abrupt Z transitions in the predicted
    # *probabilities* (not in the targets).
    probs = softmax(logits, dim=CLASS)         # over the class axis
    z_diff = probs[..., 1:, :, :, :] - probs[..., :-1, :, :, :]
    cont = z_diff.pow(2).mean()                # mean squared diff

    return alpha * ce + beta * cont
```

Config:

```yaml
loss:
  name: Cont
  weights: [3.0, 7.0]      # [bg, GBM]
  cont_alpha: 0.7          # CE term weight
  cont_beta: 0.3           # continuity term weight
```

## The class-weighted CE term

Standard PyTorch `CrossEntropyLoss(weight=tensor([3.0, 7.0]))` — each
voxel's CE contribution is multiplied by the weight of its *true* class.

The weights `[3.0, 7.0]` are heuristic (set by the project owner from
class-prevalence analysis on the source dataset). They're a soft form of
"focal loss" — pushing the gradient towards mistakes on the rare
foreground.

A small subtlety: a previous version of the loss had **buggy axes** —
it was diffing the *softmax over the class dimension* but applying it
along the *wrong axis* (mistaking class-channels for spatial-Z). Phase 1
of the cleanup fixed it; see `tests/losses/test_loss_cont.py` for the
regression tests that lock in the corrected axes (coverage ~97%).

## The continuity term

The continuity term is what makes this loss interesting for our data.
It diffs the *predicted probabilities* between consecutive Z slices and
penalises the squared difference. Concretely, for predicted
probabilities `P` of shape `(B, C, Z, H, W)`:

```python
cont = ((P[:, :, 1:, :, :] - P[:, :, :-1, :, :]) ** 2).mean()
```

This is **NOT** applied to the labels — only to the model's outputs.
The labels can be sharp (and are, after `np.repeat` upsampling); the
**predictions** should be smooth in Z, and the cont term pulls them
that way.

### What it actually penalises

The squared L2 difference between adjacent Z slices in the softmax
output. A model that predicts:
- `[bg=1.0, gbm=0.0]` at every Z position contributes **0** to cont
  (constant probabilities, zero diff).
- `[bg=0.5, gbm=0.5]` everywhere also contributes **0**.
- Hard `0 → 1` transitions in the GBM probability at sharp Z
  boundaries contribute the most.

So the term encourages "make adjacent Z slice predictions look similar
in probability space". It does NOT directly encourage "smooth out the
mask boundary in 3D" — only Z-smoothness.

### The asymmetry with the target

The target has sharp transitions (stacked-label periodicity). The
prediction is penalised for sharp transitions in probability. If the
target's sharp transition aligns with where the prediction transitions,
there's no problem — the CE term loves the alignment, the cont term
sees the prediction sliding from `[1, 0]` to `[0, 1]` over one or two
slices.

So the cont term is **conditional smoothness**: smooth where the
target allows, sharp where the target requires it. In practice this
gives the U-Net a strong push toward overall-smooth output. It gives
SwinUNETR less, because Swin can place its sharp transitions exactly
where the target's are — letting it satisfy CE and cont
simultaneously without smoothing in between. See
[the Z-jaggedness diagnosis](07-case-study-z-jaggedness/diagnosis.md).

### Why MSE on probabilities and not on logits

Two reasons:

1. Probabilities live in `[0, 1]`; MSE on probabilities saturates
   naturally. MSE on logits would have unbounded scale and dominate
   the CE term.

2. The continuity term is interpretable as: "the predicted membrane
   shouldn't move much between adjacent Z slices in physical space".
   That's most naturally expressed in probability space (probability
   of "membrane is here") rather than logit space.

## How `cont_alpha` and `cont_beta` control the tradeoff

`alpha = 0.7, beta = 0.3` means CE dominates ~2× over continuity. The
ratio matters:

- Higher `beta` → more Z smoothness, less precise boundary fitting.
  Potential cost: under-segments thin parts of the GBM.
- Higher `alpha` → tighter fit to the (stacked) target. Potential cost:
  jagged Z in the output (the SwinUNETR failure mode).

The defaults `(0.7, 0.3)` were not exhaustively tuned. They sit
comfortably in a region where the U-Net gives smooth output and
SwinUNETR gives jagged. Whether changing the ratio would close the
gap on SwinUNETR specifically is an open question — see
[future directions](07-case-study-z-jaggedness/future-directions.md).

## Other available losses

The `gbm.py` framework supports several losses:

| Name | Status |
|---|---|
| `Cont` | Active (used by all current swin and unet experiments) |
| `Dice` | Available |
| `CrossEntropy` | Plain weighted CE; no continuity term |
| `IoU` | Available |

Test coverage:
- `tests/losses/test_loss_dice.py` — 100% coverage
- `tests/losses/test_loss_cont.py` — 97%
- `tests/losses/test_loss_iou.py` — 92%

Configurable via `configs.trainer.optimization.loss.name`.

## Tests to read for understanding

If you want to convince yourself the loss really does what this doc
says, the tests are the fastest path:

- `tests/losses/test_loss_cont.py:test_cont_loss_combines_ce_and_continuity_terms`
  — locks in the additive structure.
- `tests/losses/test_loss_cont.py:test_continuity_term_is_zero_for_constant_predictions`
  — sanity check that the cont term zeros out on smooth-in-Z output.
- `tests/losses/test_loss_cont.py:test_class_weights_applied_to_cross_entropy`
  — verifies the `[3.0, 7.0]` weighting is honoured.
