"""Segmentation criterion, preserved byte-for-byte from the released `script/train_utils.py`.

The arithmetic here is load-bearing for reproducing the published IoU and is deliberately not
"cleaned up": the Dice term sums over the *whole batch* rather than per-sample, and its +1
smoothing is applied to numerator and denominator with different multiplicities. Both are
unusual, both change the number, and both are kept exactly as the authors wrote them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_loss(logits, target, bce_weight: float = 1.0, dice_weight: float = 1.0):
    """Binary cross-entropy plus a batch-level soft Dice term."""
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    probability = logits.sigmoid()
    intersection = (probability * target).sum()
    union = probability.sum() + target.sum()
    dice = 1 - (2 * intersection + 1) / (union + 1)
    return bce_weight * bce + dice_weight * dice


class SegmentationLoss(torch.nn.Module):
    """`segmentation_loss` as an instantiable module, so Hydra can carry its weights.

    A thin wrapper -- the function above stays the single implementation, and this exists only
    so `configs/task/vehicle.yaml` can set the two weights through `_target_` rather than the
    call site hardcoding them.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return segmentation_loss(
            logits, target, bce_weight=self.bce_weight, dice_weight=self.dice_weight
        )

    def extra_repr(self) -> str:
        return f"bce_weight={self.bce_weight}, dice_weight={self.dice_weight}"
