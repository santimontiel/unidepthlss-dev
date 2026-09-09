"""BEV label rasterization, shared by both dataset sources.

Extracted verbatim from the released `dataset_nuscenes.py::_build_targets` so the devkit-backed
and store-backed datasets cannot rasterize differently. `dev/check_store_parity.py` proves the
extraction and both callers against the original released code.

## The canonical box array

Both sources normalize to a `(K, 6)` float array with columns::

    [cx, cy, l, w, yaw_ego, visibility]

in the **ego-flat** frame -- origin at the LIDAR sample's ego pose with only its yaw removed,
x forward, y left. This is the frame the released code built inline
(``ego_rotation.T @ (translation - ego_translation)``) and, independently, the frame the sibling
repos' offline store already stores, which is what makes the two interchangeable.

``yaw_ego`` is the box heading in that frame (``global_yaw - ego_yaw``); the extra
``pi - yaw_ego`` below is the released code's own convention, and it is load-bearing: BEV row
and column both run *opposite* to ego x and y, and that double flip is a reflection, which
reverses the sense of rotation. Changing one without the other silently mirrors every box.
"""

from __future__ import annotations

import math

import numpy as np

from unidepthlss.geometry import BevGrid


def draw_rotated_box(mask, center_x, center_y, width, length, yaw, value):
    """Fill one rotated rectangle into `mask`. Unchanged from the release."""
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    half_width, half_length = width / 2, length / 2
    corners = []
    for x, y in (
        (-half_width, -half_length),
        (half_width, -half_length),
        (half_width, half_length),
        (-half_width, half_length),
    ):
        corners.append(
            [
                center_x + x * cos_yaw - y * sin_yaw,
                center_y + x * sin_yaw + y * cos_yaw,
            ]
        )
    points = np.round(np.asarray(corners)).astype(np.int32)
    import cv2

    cv2.fillPoly(mask, [points], value)


def rasterize_boxes(boxes: np.ndarray, grid: BevGrid) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize canonical ego-flat boxes into (segmentation, visibility) rasters.

    Returns two ``(grid.h, grid.w)`` uint8 arrays: a binary vehicle mask, and a visibility
    raster that is 255 everywhere no box covers and the box's nuScenes visibility level (0-4)
    where one does. Boxes are drawn in the order given, so later rows overwrite earlier ones
    where they overlap -- preserved from the release, and the reason the two sources' box
    ordering is compared explicitly in `dev/check_store_parity.py`.
    """
    segmentation = np.zeros((grid.h, grid.w), dtype=np.uint8)
    visibility = np.full((grid.h, grid.w), 255, dtype=np.uint8)

    for cx, cy, length, width, yaw_ego, visibility_level in boxes:
        col = (grid.half_w - cy) / grid.res_w
        row = (grid.half_h - cx) / grid.res_h
        if not (0 <= col < grid.w and 0 <= row < grid.h):
            continue

        box_args = (
            col,
            row,
            width / grid.res_w,
            length / grid.res_h,
            math.pi - yaw_ego,
        )
        draw_rotated_box(segmentation, *box_args, value=1)
        draw_rotated_box(
            visibility,
            *box_args,
            value=max(0, min(4, int(visibility_level))),
        )

    return segmentation, visibility
