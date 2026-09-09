"""Devkit-free nuScenes dataset, reading the sibling repos' existing offline store.

Same `__getitem__` contract, same rasterizer and same image pipeline as the devkit-backed
`NuScenesBEVDataset` -- proven equivalent by `dev/check_store_parity.py` and
`dev/check_dataset_parity.py`. The reason to prefer it is memory: the devkit forks an ~8 GB
in-memory index into every dataloader worker, which is the only thing capping `num_workers` at 2.
This path holds a token list and a 0.9 MB calibration array instead.

## Where each field comes from, and why

* **Sample index, image paths, intrinsics, box geometry** -- the store
  (`<dataroot>/store_det2d`), built by a sibling repo. Verified bit-identical to the devkit for
  every field this repo reads.
* **Extrinsics** -- the sidecar written by `tools/build_calibration_sidecar.py`, *not* the store.
  The store holds `cam_from_ego_flat` at the LIDAR pose; the released model consumes
  `calibrated_sensor` directly (camera-to-ego at the camera's own timestamp, pitch and roll
  kept). Measured difference: up to 0.54 m. The store's convention is arguably better, but it is
  not the one the published number was produced under, so this path reproduces the released one.

This is a label/calibration store. It holds no backbone activations -- the frozen UniDepthV2
ViT-L runs inside the network on every step, as the method requires.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from unidepthlss.data.dataset import (
    CAMERA_NAMES,
    load_and_prepare_image,
    scale_intrinsics,
)
from unidepthlss.data.rasterize import rasterize_boxes
from unidepthlss.geometry import BevGrid

#: Store classes that are NOT part of the released `category_name.startswith("vehicle")` filter.
NON_VEHICLE_CLASSES = frozenset({"pedestrian", "trafficcone", "barrier"})

SIDECAR_NAME = "unidepthlss_calibration.npz"


class NuScenesStoreDataset(Dataset):
    def __init__(
        self,
        *,
        dataroot: str,
        store_dir: str | None = None,
        calibration_sidecar: str | None = None,
        split: str = "train",
        img_size: Tuple[int, int] = (448, 798),
        grid: BevGrid | None = None,
        augment: bool | None = None,
        return_visibility: bool = False,
    ):
        self.dataroot = Path(dataroot)
        self.store_dir = Path(store_dir) if store_dir else self.dataroot / "store_det2d"
        self.img_height, self.img_width = img_size
        self.grid = grid if grid is not None else BevGrid()
        self.augment = split == "train" if augment is None else augment
        self.return_visibility = return_visibility

        header = json.loads((self.store_dir / "header.json").read_text())
        class_names = header["class_names"]
        self.vehicle_class_ids = frozenset(
            index
            for index, name in enumerate(class_names)
            if name not in NON_VEHICLE_CLASSES
        )

        # Token -> npz path. Built once here rather than per worker; it is a plain dict of
        # strings, which is cheap to fork compared with the devkit's table graph.
        self._record_paths = {
            path.stem: path
            for scene_dir in (self.store_dir / "samples").iterdir()
            if scene_dir.is_dir()
            for path in scene_dir.glob("*.npz")
        }

        split_file = self.store_dir / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"no split list at {split_file}")
        self.sample_tokens = [
            token
            for token in split_file.read_text().split()
            if token in self._record_paths
        ]

        sidecar_path = (
            Path(calibration_sidecar)
            if calibration_sidecar
            else self.dataroot / SIDECAR_NAME
        )
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"calibration sidecar not found at {sidecar_path}. Build it once with:\n"
                f"  uv run tools/build_calibration_sidecar.py --dataroot {self.dataroot}"
            )
        sidecar = np.load(sidecar_path, allow_pickle=False)
        channels = [str(c) for c in sidecar["camera_channels"]]
        if tuple(channels) != tuple(CAMERA_NAMES):
            raise ValueError(
                f"sidecar camera order {channels} does not match {list(CAMERA_NAMES)}"
            )
        self._sidecar_index = {
            str(token): position for position, token in enumerate(sidecar["tokens"])
        }
        self._extrinsics = sidecar["extrinsics"]
        self._intrinsics = sidecar["intrinsics"]
        self._image_sizes = sidecar["image_sizes"]

        missing = [t for t in self.sample_tokens if t not in self._sidecar_index]
        if missing:
            raise ValueError(
                f"{len(missing)} '{split}' tokens missing from the calibration sidecar "
                f"(first: {missing[0]}). Rebuild it for this dataset version."
            )

    def __len__(self):
        return len(self.sample_tokens)

    def _load_cameras(self, record, position: int):
        # The store writes cameras in its own order; index by channel name so this path builds
        # the rig in the released CAMERA_NAMES order regardless.
        slot_of = {
            str(channel): slot
            for slot, channel in enumerate(record["camera_channels"])
        }
        paths = record["image_paths"]

        images, intrinsics, extrinsics = [], [], []
        for camera_name in CAMERA_NAMES:
            slot = slot_of[camera_name]
            image_path = os.path.join(self.dataroot, str(paths[slot]))
            image, original_height, original_width = load_and_prepare_image(
                image_path, self.img_height, self.img_width, self.augment
            )
            camera_intrinsics = scale_intrinsics(
                record["intrinsics"][slot],
                original_height=original_height,
                original_width=original_width,
                img_height=self.img_height,
                img_width=self.img_width,
            )
            sidecar_slot = CAMERA_NAMES.index(camera_name)
            images.append(image)
            intrinsics.append(torch.from_numpy(camera_intrinsics))
            extrinsics.append(
                torch.from_numpy(self._extrinsics[position, sidecar_slot].copy())
            )

        return (
            torch.stack(images),
            torch.stack(intrinsics),
            torch.stack(extrinsics),
        )

    def _collect_boxes(self, record) -> np.ndarray:
        """Store rows -> the canonical ego-flat box array (see unidepthlss/data/rasterize.py).

        Store columns are `[cx, cy, l, w, cz, h, yaw, class, visibility, vx, vy]`, already in
        the ego-flat frame the released code builds inline -- which is what makes this a
        conversion rather than a reimplementation.
        """
        boxes = record["boxes"]
        if boxes.shape[0] == 0:
            return np.zeros((0, 6))
        keep = np.fromiter(
            (int(row) in self.vehicle_class_ids for row in boxes[:, 7]),
            dtype=bool,
            count=boxes.shape[0],
        )
        boxes = boxes[keep]
        return np.stack(
            [boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 6], boxes[:, 8]],
            axis=1,
        )

    def _build_targets(self, record):
        segmentation, visibility = rasterize_boxes(
            self._collect_boxes(record), self.grid
        )
        segmentation = torch.from_numpy(segmentation).float().unsqueeze(0)
        visibility = torch.from_numpy(visibility).long()
        return segmentation, visibility

    def __getitem__(self, index):
        token = self.sample_tokens[index]
        record = np.load(self._record_paths[token], allow_pickle=True)
        position = self._sidecar_index[token]

        images, intrinsics, extrinsics = self._load_cameras(record, position)
        segmentation, visibility = self._build_targets(record)
        output = (images, intrinsics, extrinsics, segmentation)
        if self.return_visibility:
            output += (visibility,)
        return output
