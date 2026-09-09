"""Vehicle IoU, preserved from the released `script/train_utils.py`.

**This is a dataset-level IoU, not a mean of per-batch IoUs.** TP/FP/FN accumulate across every
sample seen since the last `reset()` and the ratio is taken once at the end. Swapping it for a
per-batch mean -- the more common idiom, and what most metric libraries give you by default --
changes the reported number, so the accumulation is kept exactly as the authors wrote it.
"""

from __future__ import annotations

import torch


class BinaryIoU:
    """Accumulating TP/FP/FN over a whole split, with an optional validity mask."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    @torch.no_grad()
    def update(self, logits, target, valid_mask=None):
        prediction = logits.sigmoid() >= self.threshold
        target = target.bool()
        if valid_mask is not None:
            if valid_mask.ndim == target.ndim - 1:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.expand_as(target)
            prediction = prediction[valid_mask]
            target = target[valid_mask]
        self.true_positive += int((prediction & target).sum().item())
        self.false_positive += int((prediction & ~target).sum().item())
        self.false_negative += int((~prediction & target).sum().item())

    def compute(self):
        union = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / union if union else 1.0


class IoUMetric:
    """`BinaryIoU` plus the visibility-filter convention, so a config can select it.

    `min_visibility` reproduces the released evaluation notebook's filter. The visibility raster
    the dataset produces is 255 for background and the nuScenes visibility token (1-4) inside
    each box, so `visibility >= min_visibility` keeps all background and drops only the vehicle
    cells whose annotation falls below the threshold. `min_visibility=2` is the standard setting
    (>40% visible) and matches the sibling repos in this family; `None` disables the filter and
    reproduces the unfiltered column.
    """

    def __init__(self, threshold: float = 0.5, min_visibility: int | None = 2):
        self.min_visibility = min_visibility
        self._iou = BinaryIoU(threshold)

    def reset(self):
        self._iou.reset()

    def update(self, logits, target, visibility=None):
        mask = None
        if self.min_visibility is not None and visibility is not None:
            mask = visibility >= self.min_visibility
        self._iou.update(logits, target, valid_mask=mask)

    def compute(self):
        return self._iou.compute()
