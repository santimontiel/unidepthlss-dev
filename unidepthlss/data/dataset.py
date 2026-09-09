"""nuScenes multi-camera BEV vehicle segmentation, read live through the devkit.

Moved from the released `dataset_nuscenes.py`. The only substantive change is that the BEV
window now comes from a shared `BevGrid` instead of the local `bev_size`/`bev_res` pair, so the
label raster and the model's voxel pooling cannot describe different windows (see
`unidepthlss/geometry.py`). For the released 128 x 128 @ 0.5 m configuration the arithmetic is
identical, cell for cell.

**No backbone-feature store, ever.** Sibling repos in this family precompute a *feature* store
to disk; this repo must not. The frozen UniDepthV2 ViT-L runs inside the network on every step --
that *is* the published method -- and a ViT-L feature store over 34k samples x 6 cameras is not
affordable on this machine.

A *label/metadata* store is a different thing entirely, and is worth using: see
`store_dataset.py`, which reads the sibling repos' existing nuScenes store instead of the devkit.
It holds raw box geometry, not rasterized masks, so both paths run the same rasterizer and
produce the same labels -- proven by `dev/check_store_parity.py`. The reason to prefer it is
memory: this devkit-backed path forks an ~8 GB in-memory index into every dataloader worker,
which is what caps `num_workers` at 2 here.
"""

from __future__ import annotations

import math
import os
import random
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from unidepthlss.data.rasterize import rasterize_boxes
from unidepthlss.geometry import BevGrid

CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
CLASS_NAMES = ("vehicle",)


def load_and_prepare_image(image_path, img_height: int, img_width: int, augment: bool):
    """Read one camera image, resize it, and optionally augment. Unchanged from the release.

    Returns the CHW float tensor in [0, 1] plus the original (height, width), which the caller
    needs to rescale that camera's intrinsics to the resized frame.

    Shared by both dataset sources so the image pipeline cannot drift between them. The
    augmentation draws from the global `random` stream, once per camera, in this order -- the
    sequence is part of what a seeded run reproduces.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    image = image[:, :, ::-1]
    original_height, original_width = image.shape[:2]
    image = cv2.resize(image, (img_width, img_height))
    image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255

    if augment:
        if random.random() < 0.8:
            brightness, contrast, saturation = [
                1 + random.uniform(-0.3, 0.3) for _ in range(3)
            ]
            hue = random.uniform(-0.1, 0.1)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            image = TF.adjust_hue(image, hue)
        if random.random() < 0.5:
            image = (image + torch.randn_like(image) * 0.02).clamp(0, 1)

    return image, original_height, original_width


def scale_intrinsics(camera_intrinsics, *, original_height, original_width,
                     img_height: int, img_width: int):
    """Rescale a 3x3 intrinsic matrix to the resized image. Unchanged from the release."""
    camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float32).copy()
    scale_x = img_width / original_width
    scale_y = img_height / original_height
    camera_intrinsics[0, 0] *= scale_x
    camera_intrinsics[0, 2] *= scale_x
    camera_intrinsics[1, 1] *= scale_y
    camera_intrinsics[1, 2] *= scale_y
    return camera_intrinsics


class NuScenesBEVDataset(Dataset):
    def __init__(
        self,
        *,
        dataroot: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        img_size: Tuple[int, int] = (294, 518),
        grid: BevGrid | None = None,
        augment: bool | None = None,
        return_visibility: bool = False,
    ):
        self.img_height, self.img_width = img_size
        self.grid = grid if grid is not None else BevGrid()
        self.augment = split == "train" if augment is None else augment
        self.return_visibility = return_visibility
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        split_name = f"mini_{split}" if "mini" in version else split
        scene_names = set(create_splits_scenes()[split_name])
        scenes = [scene for scene in self.nusc.scene if scene["name"] in scene_names]
        if not scenes and version == "v1.0-test" and split == "test":
            scenes = self.nusc.scene

        self.sample_tokens = []
        for scene in scenes:
            token = scene["first_sample_token"]
            while token:
                self.sample_tokens.append(token)
                token = self.nusc.get("sample", token)["next"]

    def __len__(self):
        return len(self.sample_tokens)

    def _load_cameras(self, sample):
        images, intrinsics, extrinsics = [], [], []
        for camera_name in CAMERA_NAMES:
            sample_data = self.nusc.get(
                "sample_data", sample["data"][camera_name]
            )
            image_path = os.path.join(self.nusc.dataroot, sample_data["filename"])
            image, original_height, original_width = load_and_prepare_image(
                image_path, self.img_height, self.img_width, self.augment
            )

            calibration = self.nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"]
            )
            camera_intrinsics = scale_intrinsics(
                calibration["camera_intrinsic"],
                original_height=original_height,
                original_width=original_width,
                img_height=self.img_height,
                img_width=self.img_width,
            )

            transform = torch.eye(4, dtype=torch.float32)
            transform[:3, :3] = torch.from_numpy(
                Quaternion(calibration["rotation"]).rotation_matrix
            ).float()
            transform[:3, 3] = torch.tensor(
                calibration["translation"], dtype=torch.float32
            )

            images.append(image)
            intrinsics.append(torch.from_numpy(camera_intrinsics))
            extrinsics.append(transform)

        return (
            torch.stack(images),
            torch.stack(intrinsics),
            torch.stack(extrinsics),
        )

    def _collect_boxes(self, sample):
        """Vehicle annotations for one sample, as canonical ego-flat boxes.

        The ego-flat transform here is the released code's, unchanged: translate by the LIDAR
        sample's ego translation, then undo only the yaw of its rotation. `unidepthlss/data/
        rasterize.py` documents the resulting column layout.

        The class filter is likewise the released one -- `category_name.split(".")[0] ==
        "vehicle"`, which is nuScenes' car / truck / bus / trailer / construction / motorcycle /
        bicycle / emergency, and excludes pedestrians and movable objects.
        """
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_yaw = Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]
        ego_rotation = Quaternion(
            scalar=np.cos(ego_yaw / 2),
            vector=[0, 0, np.sin(ego_yaw / 2)],
        ).rotation_matrix
        ego_translation = np.asarray(ego_pose["translation"])

        boxes = []
        for annotation_token in sample["anns"]:
            annotation = self.nusc.get("sample_annotation", annotation_token)
            if annotation["category_name"].split(".")[0] != "vehicle":
                continue

            position = ego_rotation.T @ (
                np.asarray(annotation["translation"]) - ego_translation
            )
            width, length, _ = annotation["size"]
            global_yaw = Quaternion(annotation["rotation"]).yaw_pitch_roll[0]

            visibility_token = annotation.get("visibility_token", "0")
            try:
                visibility_level = int(visibility_token)
            except (TypeError, ValueError):
                visibility_level = 0

            boxes.append(
                [
                    position[0],
                    position[1],
                    length,
                    width,
                    global_yaw - ego_yaw,
                    visibility_level,
                ]
            )

        return np.asarray(boxes, dtype=np.float64).reshape(-1, 6)

    def _build_targets(self, sample):
        segmentation, visibility = rasterize_boxes(
            self._collect_boxes(sample), self.grid
        )
        segmentation = torch.from_numpy(segmentation).float().unsqueeze(0)
        visibility = torch.from_numpy(visibility).long()
        return segmentation, visibility

    def __getitem__(self, index):
        sample = self.nusc.get("sample", self.sample_tokens[index])
        images, intrinsics, extrinsics = self._load_cameras(sample)
        segmentation, visibility = self._build_targets(sample)
        output = (images, intrinsics, extrinsics, segmentation)
        if self.return_visibility:
            output += (visibility,)
        return output
