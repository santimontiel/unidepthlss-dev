"""Write the camera extrinsics the released code uses, so training never needs the devkit.

    uv run tools/build_calibration_sidecar.py
    uv run tools/build_calibration_sidecar.py --version v1.0-mini --split-file-prefix mini

The sibling repos' nuScenes store already carries this repo's labels, intrinsics and image paths
bit-identically (proven by `dev/check_store_parity.py`), but *not* its extrinsics: the store
holds `cam_from_ego_flat` at the LIDAR pose, whereas the released model consumes
`calibrated_sensor` directly -- camera-to-ego at the camera's own timestamp, pitch and roll
kept. Measured over 200 val samples the two differ by up to 0.54 m, so adopting the store's
version would move the published numbers.

This tool closes that one gap. It walks the devkit once and writes a single small npz of
released-convention extrinsics keyed by sample token (~13 MB for the full trainval split, six
4x4 float32 matrices per sample). After it has run, `NuScenesStoreDataset` reads the store plus
this file and the devkit is never imported at training time -- which is what lets `num_workers`
rise from 2 to something useful, since the devkit's ~8 GB in-memory index was being forked into
every worker.

This is a *calibration* sidecar, not a feature store. It is kilobytes per sample and holds no
backbone activations; the frozen UniDepthV2 ViT-L stays inside the network, as the method
requires.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from unidepthlss.data.dataset import CAMERA_NAMES  # noqa: E402

SIDECAR_NAME = "unidepthlss_calibration.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="/data/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--out",
        default=None,
        help=f"output npz (default: <dataroot>/{SIDECAR_NAME})",
    )
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    out = Path(args.out) if args.out else dataroot / SIDECAR_NAME

    from nuscenes import NuScenes
    from pyquaternion import Quaternion

    print(f"loading devkit {args.version} from {dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)

    tokens: list[str] = []
    extrinsics: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    image_sizes: list[np.ndarray] = []

    for count, sample in enumerate(nusc.sample):
        per_camera_e = np.zeros((len(CAMERA_NAMES), 4, 4), dtype=np.float32)
        per_camera_k = np.zeros((len(CAMERA_NAMES), 3, 3), dtype=np.float32)
        per_camera_size = np.zeros((len(CAMERA_NAMES), 2), dtype=np.int32)

        for slot, channel in enumerate(CAMERA_NAMES):
            sample_data = nusc.get("sample_data", sample["data"][channel])
            calibration = nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"]
            )
            # Exactly what NuScenesBEVDataset._load_cameras builds: camera-to-ego, full
            # rotation, no ego-flat de-yawing, no timestamp compensation.
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = Quaternion(calibration["rotation"]).rotation_matrix
            transform[:3, 3] = calibration["translation"]
            per_camera_e[slot] = transform
            per_camera_k[slot] = np.asarray(
                calibration["camera_intrinsic"], dtype=np.float32
            )
            # Original resolution, needed to rescale the intrinsics to the training resize.
            per_camera_size[slot] = (sample_data["height"], sample_data["width"])

        tokens.append(sample["token"])
        extrinsics.append(per_camera_e)
        intrinsics.append(per_camera_k)
        image_sizes.append(per_camera_size)

        if (count + 1) % 5000 == 0:
            print(f"  {count + 1} / {len(nusc.sample)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tokens=np.asarray(tokens),
        camera_channels=np.asarray(CAMERA_NAMES),
        extrinsics=np.stack(extrinsics),
        intrinsics=np.stack(intrinsics),
        image_sizes=np.stack(image_sizes),
    )
    size_mb = out.stat().st_size / 1e6
    print(f"\nwrote {out}  ({len(tokens):,} samples, {size_mb:.1f} MB)")
    print("camera order:", ", ".join(CAMERA_NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
