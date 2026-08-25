"""Training losses for the codec."""

from losses.contact_losses import contact_bce_loss, contact_focal_loss
from losses.umr_loss import UMRLoss, UMRLossOutput, UMRLossWeights

__all__ = [
    "UMRLoss",
    "UMRLossOutput",
    "UMRLossWeights",
    "contact_bce_loss",
    "contact_focal_loss",
]
