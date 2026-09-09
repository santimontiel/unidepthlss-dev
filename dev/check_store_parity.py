"""Prove what of the sibling repos' nuScenes store this repo can reuse, and what it cannot.

The store at `<nuscenes>/store_det2d` was built by a sibling repo (tinycar-dev) and already
holds, per sample, the raw annotation geometry and camera calibration this repo needs. Reusing
it removes the nuScenes devkit from the training path entirely -- the devkit forks an ~8 GB
in-memory index into every dataloader worker, which is the only reason `num_workers` is capped
at 2 here.

But "already has the data" is not the same as "has it in the convention the published numbers
were produced under". This script settles that empirically, per field, against the devkit:

    uv run dev/check_store_parity.py --limit 200

It checks three things and treats them differently:

* **Labels.** Both paths must produce byte-identical segmentation rasters. This is the gate: the
  store keeps raw box geometry (not rasterized masks), and `unidepthlss/data/rasterize.py` is
  shared, so any difference here is a real convention mismatch, not a rounding artefact.
* **Intrinsics.** Expected to match exactly.
* **Extrinsics.** Expected NOT to match, and that is the point. The store holds
  `cam_from_ego_flat` -- ego-flat at the LIDAR sample's pose. The released code uses
  `calibrated_sensor` directly, i.e. camera-to-ego at the *camera's own* timestamp, with pitch
  and roll kept. The two differ by the ego motion between the camera and LIDAR captures plus the
  dropped pitch/roll. The store's convention is arguably the more correct one, but it is not the
  one the published 49.4 IoU was produced under, so the store-backed dataset must not adopt it
  -- see `tools/build_calibration_sidecar.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from unidepthlss.data.rasterize import rasterize_boxes  # noqa: E402
from unidepthlss.geometry import BevGrid  # noqa: E402

# Store class indices that make up the released code's `category_name.startswith("vehicle")`
# filter: everything except pedestrian, trafficcone and barrier.
NON_VEHICLE_CLASSES = {"pedestrian", "trafficcone", "barrier"}


def store_boxes(record, class_names) -> np.ndarray:
    """Canonical ego-flat boxes from a store record. See unidepthlss/data/rasterize.py."""
    boxes = record["boxes"]
    if boxes.shape[0] == 0:
        return np.zeros((0, 6))
    vehicle_ids = {
        index
        for index, name in enumerate(class_names)
        if name not in NON_VEHICLE_CLASSES
    }
    keep = np.array([int(row[7]) in vehicle_ids for row in boxes], dtype=bool)
    boxes = boxes[keep]
    # store columns: [cx, cy, l, w, cz, h, yaw, class, visibility, vx, vy]
    return np.stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 6], boxes[:, 8]],
        axis=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="/data/nuscenes")
    parser.add_argument("--store", default=None, help="defaults to <dataroot>/store_det2d")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    store = Path(args.store) if args.store else dataroot / "store_det2d"
    header = json.loads((store / "header.json").read_text())
    class_names = header["class_names"]

    grid = BevGrid(h=200, w=200, h_meters=100.0, w_meters=100.0)
    print(f"grid: {grid.describe()}\n")

    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    print("loading devkit (this is the cost the store exists to remove)...")
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)

    from unidepthlss.data.dataset import NuScenesBEVDataset

    devkit = NuScenesBEVDataset.__new__(NuScenesBEVDataset)
    devkit.nusc = nusc
    devkit.grid = grid

    index = {}
    for scene_dir in (store / "samples").iterdir():
        for path in scene_dir.glob("*.npz"):
            index[path.stem] = path

    store_tokens = (store / "splits" / f"{args.split}.txt").read_text().split()

    # The split lists must name the SAME samples, not merely overlap: a store split that is a
    # subset would still pass every per-sample comparison below while quietly evaluating fewer
    # samples than the published number was computed over.
    scene_names = set(create_splits_scenes()[args.split])
    devkit_tokens = []
    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue
        token = scene["first_sample_token"]
        while token:
            devkit_tokens.append(token)
            token = nusc.get("sample", token)["next"]
    same_split = set(devkit_tokens) == set(store_tokens)
    print(f"split '{args.split}': devkit {len(devkit_tokens)}, store {len(store_tokens)}, "
          f"identical set: {same_split}")
    if not same_split:
        print("❌ split lists differ -- the store is NOT a drop-in for evaluation")
        return 1

    tokens = [t for t in store_tokens if t in index][: args.limit]
    print(f"comparing {len(tokens)} '{args.split}' samples\n")

    seg_mismatch = vis_mismatch = 0
    box_count_mismatch = 0
    intr_max = 0.0
    extr_max = 0.0

    for token in tokens:
        sample = nusc.get("sample", token)
        record = np.load(index[token], allow_pickle=True)

        a = devkit._collect_boxes(sample)
        b = store_boxes(record, class_names)
        if a.shape[0] != b.shape[0]:
            box_count_mismatch += 1

        seg_a, vis_a = rasterize_boxes(a, grid)
        seg_b, vis_b = rasterize_boxes(b, grid)
        seg_mismatch += int(not np.array_equal(seg_a, seg_b))
        vis_mismatch += int(not np.array_equal(vis_a, vis_b))

        # Calibration, in the store's own camera order.
        for slot, channel in enumerate(record["camera_channels"]):
            sd = nusc.get("sample_data", sample["data"][str(channel)])
            cal = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            intr_max = max(
                intr_max,
                float(
                    np.abs(
                        np.asarray(cal["camera_intrinsic"]) - record["intrinsics"][slot]
                    ).max()
                ),
            )
            from pyquaternion import Quaternion

            ego_from_cam = np.eye(4)
            ego_from_cam[:3, :3] = Quaternion(cal["rotation"]).rotation_matrix
            ego_from_cam[:3, 3] = cal["translation"]
            store_ego_from_cam = np.linalg.inv(record["extrinsics"][slot].astype(np.float64))
            extr_max = max(extr_max, float(np.abs(ego_from_cam - store_ego_from_cam).max()))

    n = len(tokens)
    print("LABELS (the gate -- must be identical)")
    print(f"  box-count mismatches      : {box_count_mismatch} / {n}")
    print(f"  segmentation mismatches   : {seg_mismatch} / {n}")
    print(f"  visibility mismatches     : {vis_mismatch} / {n}")
    print()
    print("CALIBRATION")
    print(f"  max |intrinsics diff|     : {intr_max:.3e}   (expect ~0)")
    print(f"  max |extrinsics diff|     : {extr_max:.3e}   (expect NON-zero: different convention)")
    print()

    ok = seg_mismatch == 0 and box_count_mismatch == 0 and intr_max < 1e-6
    if ok:
        print("✅ labels + intrinsics are reusable from the store.")
        if vis_mismatch:
            print(f"   NOTE: {vis_mismatch} visibility rasters differ -- box draw order differs")
            print("   (the store groups by class), and overlapping boxes overwrite. Affects only")
            print("   the filtered IoU column, and only in overlap regions.")
        print("❗ extrinsics are NOT reusable -- build the sidecar instead:")
        print("   uv run tools/build_calibration_sidecar.py")
    else:
        print("❌ store is NOT a drop-in for labels; do not switch the default.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
