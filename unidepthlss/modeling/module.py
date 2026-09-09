"""The Lightning training loop, ported from the released `script/train_utils.py::run_epoch`.

The port replaces the hand-rolled loop's manual `.item()` accumulation, fp16 `GradScaler`, and
`torch.save` calls with Lightning's own machinery. Three things are deliberately carried over
unchanged because they decide the reported number:

* **AdamW + `CosineAnnealingLR(T_max=max_epochs, eta_min=1e-6)`, stepped per epoch.** The paper
  names this schedule explicitly; it is not swapped for the family's usual OneCycleLR.
* **The optimizer's parameter filter, `p for p in model.parameters() if p.requires_grad`.** With
  the default (faithful) `strict_freeze=False` this includes 1.4M backbone tensors that
  `DinoVisionTransformer.train()` keeps re-enabling -- see
  `LiftSplatProjector._enforce_freeze`. That is what the released training actually did.
* **Dataset-level IoU** (`unidepthlss/metrics.py`), accumulated across the split rather than
  averaged per batch.

Metrics are updated on rank zero's view of each batch and reduced at epoch end via
`self.log(..., sync_dist=True)`, so a multi-GPU run reports the true global IoU rather than a
rank-0-skewed one.
"""

from __future__ import annotations

from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn

from unidepthlss.losses import SegmentationLoss
from unidepthlss.metrics import IoUMetric


class UniDepthLSSModule(L.LightningModule):
    def __init__(
        self,
        cfg: Any = None,
        model: nn.Module = None,
        losses: Any = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        eta_min: float = 1e-6,
        max_epochs: int = 30,
        head_bias_init: float | None = -2.19,
        iou_threshold: float = 0.5,
        min_visibility: int = 2,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.losses = losses if losses is not None else SegmentationLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.max_epochs = max_epochs
        self.min_visibility = min_visibility

        if head_bias_init is not None:
            self.model.initialize_head_bias(head_bias_init)

        # Two metrics, one pass: the unfiltered and the visibility-filtered column of the paper's
        # Table 1. Plain objects rather than a ModuleDict -- these hold Python int counters, not
        # tensors, so there is no device state for Lightning to move.
        #
        # **Kept per stage, and that is load-bearing.** Lightning runs the validation loop *inside*
        # the training epoch, before `on_train_epoch_end`. With a single shared pair the order
        # becomes: train batches accumulate -> val batches accumulate into the same counters ->
        # `on_validation_epoch_end` flushes and resets -> `on_train_epoch_end` flushes counters
        # that are now empty. The observed symptom is `train/metrics/iou == 1.0000` exactly (the
        # `union == 0 -> 1.0` fallback in BinaryIoU.compute), and, worse and silently, a
        # `val/metrics/iou` polluted with training batches -- which is the value ModelCheckpoint
        # monitors.
        self.metrics = {
            stage: {
                "iou": IoUMetric(threshold=iou_threshold, min_visibility=None),
                "iou_visible": IoUMetric(
                    threshold=iou_threshold, min_visibility=min_visibility
                ),
            }
            for stage in ("train", "val")
        }

    def forward(self, batch: Dict[str, Any]):
        return self.model(batch["images"], batch["intrinsics"], batch["extrinsics"])

    def common_step(self, batch: Dict[str, Any], stage: str = "train") -> Dict[str, Any]:
        target = batch["segmentation"]
        batch_size = target.shape[0]

        logits = self(batch)
        loss = self.losses(logits, target)

        self.log(
            f"{stage}/loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        visibility = batch.get("visibility")
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), target, visibility=visibility)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, stage="val")

    def _flush_metrics(self, stage: str) -> None:
        for name, metric in self.metrics[stage].items():
            self.log(
                f"{stage}/metrics/{name}",
                torch.tensor(metric.compute(), device=self.device),
                on_epoch=True,
                logger=True,
                prog_bar=(name == "iou"),
                sync_dist=True,
            )
            metric.reset()

    def on_train_epoch_end(self) -> None:
        self._flush_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._flush_metrics("val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
