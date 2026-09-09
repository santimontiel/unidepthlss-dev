"""BEV segmentation head. Moved unchanged from the released `Model/model.py`.

Named `SegmentationHead` here (it was `_SegmentationHead`); the class name is not part of the
checkpoint's state-dict keys, which are built from attribute names -- `seg.net.0.weight` and so
on -- and those are unchanged.
"""

from __future__ import annotations

import torch.nn as nn


class SegmentationHead(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, 1),
        )

    def forward(self, features):
        return self.net(features)
