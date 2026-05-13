# Library Imports
import torch
import torch.nn as nn


def continuity_loss_diff(logits):
    # logits shape: (B, C, D, H, W) — softmax over the class dim, diff along depth.
    assert logits.dim() == 5, "Logits should be a 5D tensor (B, C, D, H, W)"

    probs = torch.softmax(logits, dim=1)

    probs_current = probs[:, :, :-1, :, :]
    probs_next = probs[:, :, 1:, :, :]

    return torch.abs(probs_current - probs_next).mean()


class ContLoss(nn.Module):
    """``alpha * CE + beta * continuity`` with **fixed** weights.

    A2-precursor refactor: the previous version learned ``alpha``/``beta``
    via sigmoid-renormalise-with-floors on two ``nn.Parameter``s, with an
    L2 regulariser on the pre-sigmoid weights. That mechanism was opaque
    — the gradient on the raw parameters was dominated by sigmoid
    saturation, and the L2 reg fired on weights that had a non-monotone
    relationship to the effective ``alpha``/``beta``. Replaced with a
    single fixed weight per term, sweepable as a hyperparameter.

    Defaults (``alpha=0.7``, ``beta=0.3``) approximate the initialisation-
    time equilibrium of the prior learned mechanism so existing experiments
    don't see a sudden behaviour shift.
    """

    def __init__(self,
                 cross_entropy_loss,
                 _alpha: float = 0.7,
                 _beta: float = 0.3):
        super().__init__()
        self.cross_entropy_loss = cross_entropy_loss
        self.alpha = float(_alpha)
        self.beta = float(_beta)

    def forward(self, logits, labels):
        ce_loss = self.cross_entropy_loss(logits, labels)
        cont_loss = continuity_loss_diff(logits)
        return self.alpha * ce_loss + self.beta * cont_loss
