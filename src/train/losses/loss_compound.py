"""Weighted sum of base segmentation losses.

Enables configs like ``Dice + CrossEntropy`` (the catalog's A2 (b)) or
``Dice + Cont`` (A2 (c)) without writing a bespoke class per combination.

Config form (consumed by ``Factory.createLoss``):

    trainer:
      loss: Compound
      compound_loss:
        - {name: Dice, weight: 1.0}
        - {name: CrossEntropy, weight: 1.0}
        - {name: Cont, weight: 0.5, params: {cont_alpha: 1.0, cont_beta: 0.0}}

Each entry has a ``name`` (one of the registered single-loss names),
an optional ``weight`` (default 1.0), and an optional ``params`` dict
of constructor kwargs forwarded to the sub-loss builder. Compound losses
cannot nest (no ``Compound`` inside ``compound_loss``) — flatten if
you need that.
"""

from collections.abc import Sequence

from torch import Tensor, nn


class CompoundLoss(nn.Module):
    """``sum(w_i * loss_i(logits, labels))`` over the supplied components.

    Components live on a ``nn.ModuleList`` so they participate in the
    parent's ``.to(device)`` calls and ``.parameters()`` enumeration —
    important when one of the sub-losses has its own learnable bits
    (currently none of the project's losses do, but ContLoss had them
    pre-A2-precursor).
    """

    def __init__(self, components: Sequence[tuple[nn.Module, float]]):
        super().__init__()
        if not components:
            raise ValueError("CompoundLoss needs at least one component.")
        self.sub_losses = nn.ModuleList([c[0] for c in components])
        self.weights = [float(c[1]) for c in components]

    def forward(self, logits, labels) -> Tensor:
        total = None
        for sub_loss, w in zip(self.sub_losses, self.weights):
            term = w * sub_loss(logits, labels)
            total = term if total is None else total + term
        return total
