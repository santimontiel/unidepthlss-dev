"""Lightning DataModule over `NuScenesBEVDataset`.

The dataset's `__getitem__` contract is preserved exactly as released -- it returns a tuple, not
a dict. Rather than change that load-bearing code, the adaptation happens here in the collate:
one place, no effect on what the dataset itself computes.
"""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from unidepthlss.data.dataset import NuScenesBEVDataset
from unidepthlss.data.store_dataset import NuScenesStoreDataset
from unidepthlss.geometry import BevGrid

#: Field order produced by `NuScenesBEVDataset.__getitem__`, with `visibility` present only when
#: the dataset was built with `return_visibility=True`.
BATCH_FIELDS = ("images", "intrinsics", "extrinsics", "segmentation", "visibility")


def collate_to_dict(samples):
    """Stack a list of dataset tuples into a named batch dict."""
    stacked = default_collate(samples)
    return dict(zip(BATCH_FIELDS, stacked))


def _normalize_img_size(img_size) -> tuple[int, int]:
    """Accept `{h: 448, w: 798}` or `(448, 798)` and return plain ints.

    The config expresses this as a mapping so other nodes can interpolate `${data.img_size.h}`
    into the model. `tuple()` on a mapping yields its *keys*, which reaches cv2.resize as
    ('h', 'w') and fails there rather than here -- so the conversion is explicit, and the ints
    are cast out of their OmegaConf wrappers on the way through.
    """
    if isinstance(img_size, Mapping):
        return int(img_size["h"]), int(img_size["w"])
    height, width = img_size
    return int(height), int(width)


class NuScenesDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        dataroot: str,
        version: str = "v1.0-trainval",
        source: str = "store",
        store_dir: str | None = None,
        calibration_sidecar: str | None = None,
        img_size: tuple[int, int] = (448, 798),
        grid: BevGrid | None = None,
        batch_size: int = 2,
        num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = 2,
        augment_train: bool = True,
        debug_mode: bool = False,
        # Carrier keys. They live under `data` in the config so other nodes can interpolate them
        # (`${data.bev}` reaches the model's grid, `${data.depth_max}` its depth clamp), which
        # means Hydra hands them to this constructor too. Accepted and ignored rather than
        # consumed -- `grid` above is the one this class actually uses, and it is interpolated
        # from `bev`, so accepting `bev` here as well would give the same value two names.
        bev: object = None,
        depth_max: float | None = None,
        dataset_name: str | None = None,
    ) -> None:
        super().__init__()
        self.dataroot = dataroot
        self.version = version
        if source not in ("store", "devkit"):
            raise ValueError(f"data.source must be 'store' or 'devkit', got {source!r}")
        self.source = source
        self.store_dir = store_dir
        self.calibration_sidecar = calibration_sidecar
        self.img_size = _normalize_img_size(img_size)
        self.grid = grid if grid is not None else BevGrid()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # Both are meaningless (and persistent_workers is an error) with a single-process loader.
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.augment_train = augment_train
        self.debug_mode = debug_mode
        self.train_dataset = None
        self.val_dataset = None

    def _build(self, split: str, augment: bool):
        # `return_visibility` is always on. The visibility raster is one uint8 map per sample,
        # and carrying it unconditionally is what lets a single eval pass report both the
        # filtered and the unfiltered IoU column -- the two settings the paper's Table 1
        # compares against.
        #
        # The two sources are verified interchangeable by dev/check_dataset_parity.py (identical
        # images, intrinsics, extrinsics, segmentation and visibility). `store` is the default
        # purely for memory: `devkit` forks an ~8 GB in-memory index into every worker, which is
        # what forces num_workers down to 2.
        if self.source == "store":
            return NuScenesStoreDataset(
                dataroot=self.dataroot,
                store_dir=self.store_dir,
                calibration_sidecar=self.calibration_sidecar,
                split=split,
                img_size=self.img_size,
                grid=self.grid,
                augment=augment,
                return_visibility=True,
            )
        return NuScenesBEVDataset(
            dataroot=self.dataroot,
            version=self.version,
            split=split,
            img_size=self.img_size,
            grid=self.grid,
            augment=augment,
            return_visibility=True,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build("train", augment=self.augment_train)
        if stage in (None, "fit", "validate", "test"):
            self.val_dataset = self._build("val", augment=False)

    def _loader(self, dataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=collate_to_dict,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False, drop_last=False)
