"""Foot-contact classification losses.

Contact predictions occupy the trailing 4 dims of the 499D UMR record
(``SLICE_FOOT_CONTACT``, order [LeftFoot, LeftToeBase, RightFoot, RightToeBase]);
the decoder emits raw logits. A focal-BCE variant handles heavily imbalanced
contacts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

def contact_bce_loss(
    pred_contact_logits: Tensor,                 # [B, T, 4]
    gt_foot_contact: Tensor,                     # [B, T, 4] in {0, 1} or [0, 1]
    *,
    pos_weight: Tensor | float | None = None,
) -> Tensor:
    """Standard BCE-with-logits on the four foot-contact channels.

    ``pos_weight`` (optional) follows :func:`torch.nn.functional.binary_cross_entropy_with_logits`
    semantics: a per-channel scalar tensor ``[4]`` (or scalar) that
    upweights the positive class when contact is rare.
    """
    if pred_contact_logits.shape != gt_foot_contact.shape:
        raise ValueError(
            f"pred / gt contact shapes must match; got "
            f"{tuple(pred_contact_logits.shape)} vs {tuple(gt_foot_contact.shape)}"
        )
    if pred_contact_logits.shape[-1] != 4:
        raise ValueError(
            f"contact last dim must be 4; got {tuple(pred_contact_logits.shape)}"
        )
    pw: Tensor | None
    if pos_weight is None:
        pw = None
    elif isinstance(pos_weight, Tensor):
        pw = pos_weight.to(pred_contact_logits.device, pred_contact_logits.dtype)
    else:
        pw = torch.full((4,), float(pos_weight), device=pred_contact_logits.device, dtype=pred_contact_logits.dtype)
    return F.binary_cross_entropy_with_logits(
        pred_contact_logits,
        gt_foot_contact.detach().to(pred_contact_logits.dtype),
        pos_weight=pw,
    )

def contact_focal_loss(
    pred_contact_logits: Tensor,                 # [B, T, 4]
    gt_foot_contact: Tensor,                     # [B, T, 4]
    *,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> Tensor:
    """Focal BCE for imbalanced foot contact.

    ``(1 - p_t)^gamma`` upweights hard examples; ``alpha`` weights the positive
    class. At ``alpha=0.5, gamma=0`` it equals ``0.5 * BCE``.
    """
    if pred_contact_logits.shape != gt_foot_contact.shape:
        raise ValueError(
            f"pred / gt contact shapes must match; got "
            f"{tuple(pred_contact_logits.shape)} vs {tuple(gt_foot_contact.shape)}"
        )
    target = gt_foot_contact.detach().to(pred_contact_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(pred_contact_logits, target, reduction="none")
    p = torch.sigmoid(pred_contact_logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = alpha_t * (1.0 - p_t).pow(gamma) * bce
    return focal.mean()
