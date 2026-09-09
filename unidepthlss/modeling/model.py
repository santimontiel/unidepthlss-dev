"""UniDepth-LSS: pure `nn.Module` forward wiring.

Moved from the released `Model/model.py`. Training-loop concerns live in `module.py`; this file
holds only the network. The `LiftSplatProjector` and `SegmentationHead` blocks moved into
`components/`.
"""

from __future__ import annotations

import torch.nn as nn

from unidepthlss.geometry import BevGrid
from unidepthlss.modeling.components import LiftSplatProjector, SegmentationHead


class UniDepthLSS(nn.Module):
    """UniDepth V2 features and metric depth projected into a BEV grid."""

    def __init__(
        self,
        *,
        img_height: int = 294,
        img_width: int = 518,
        num_classes: int = 1,
        feature_channels: int = 128,
        grid: BevGrid | None = None,
        **projector_kwargs,
    ):
        super().__init__()
        self.projector = LiftSplatProjector(
            img_height=img_height,
            img_width=img_width,
            feature_channels=feature_channels,
            grid=grid,
            **projector_kwargs,
        )
        self.seg = SegmentationHead(feature_channels, num_classes)

    @property
    def grid(self) -> BevGrid:
        return self.projector.grid

    def freeze_backbone(self, freeze: bool = True):
        self.projector.backbone.requires_grad_(not freeze)
        if freeze:
            # Without this the freeze is undone the next time anything calls .train()/.eval()
            # on the backbone -- see LiftSplatProjector._enforce_freeze.
            self.projector.strict_freeze = True
            self.projector._enforce_freeze()

    def initialize_head_bias(self, bias: float = -2.19):
        nn.init.constant_(self.seg.net[-1].bias, bias)

    def forward(self, images, intrinsics, extrinsics):
        return self.seg(self.projector(images, intrinsics, extrinsics))
