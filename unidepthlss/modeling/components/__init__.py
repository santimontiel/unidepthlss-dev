"""Interchangeable architectural blocks, re-exported so Hydra `_target_` paths stay short."""

from unidepthlss.modeling.components.projector import LiftSplatProjector
from unidepthlss.modeling.components.seg_head import SegmentationHead

__all__ = ["LiftSplatProjector", "SegmentationHead"]
