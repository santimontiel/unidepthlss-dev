"""Write the camera extrinsics the released code uses, so training never needs the devkit.

    uv run tools/build_calibration_sidecar.py            # build if absent
    uv run tools/build_calibration_sidecar.py --check    # verify only, build nothing
    uv run tools/build_calibration_sidecar.py --force    # rebuild in place

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
from typing import NamedTuple

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from unidepthlss.data.dataset import CAMERA_NAMES  # noqa: E402

SIDECAR_NAME = "unidepthlss_calibration.npz"


class SidecarStatus(NamedTuple):
    ok: bool
    message: str


def verify(out: Path, store_dir: Path) -> SidecarStatus:
    """Check a sidecar without loading the devkit.

    Deliberately validated against the *store's* split lists rather than by re-walking nuScenes:
    the devkit costs ~8 GB and half a minute to open, which would make this guard too expensive
    to run in front of every training job -- which is exactly where it earns its keep. When the
    store is absent the coverage check is skipped rather than failed, since `data.source=devkit`
    is a legitimate configuration that needs no store.
    """
    if not out.is_file():
        return SidecarStatus(False, f"\u274c missing    {out}")

    try:
        sidecar = np.load(out, allow_pickle=False)
        tokens = set(str(t) for t in sidecar["tokens"])
        channels = [str(c) for c in sidecar["camera_channels"]]
    except Exception as exc:  # noqa: BLE001 -- a corrupt or truncated file must read as "rebuild"
        return SidecarStatus(False, f"\u274c unreadable {out} ({type(exc).__name__}: {exc})")

    if tuple(channels) != tuple(CAMERA_NAMES):
        return SidecarStatus(
            False,
            f"\u274c camera order {channels} != {list(CAMERA_NAMES)} in {out}",
        )

    splits_dir = store_dir / "splits"
    if not splits_dir.is_dir():
        return SidecarStatus(
            True,
            f"\u2705 sidecar    {out} ({len(tokens):,} samples)\n"
            f"\u2139\ufe0f  no store at {store_dir}; coverage not checked "
            f"(fine for data.source=devkit)",
        )

    missing_total = 0
    for split_file in sorted(splits_dir.glob("*.txt")):
        needed = set(split_file.read_text().split())
        missing = needed - tokens
        missing_total += len(missing)
        if missing:
            return SidecarStatus(
                False,
                f"\u274c {out} is missing {len(missing):,} of {len(needed):,} "
                f"'{split_file.stem}' tokens (first: {sorted(missing)[0]})",
            )

    return SidecarStatus(
        True, f"\u2705 sidecar    {out} ({len(tokens):,} samples, covers every split)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="/data/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--out",
        default=None,
        help=f"output npz (default: <dataroot>/{SIDECAR_NAME})",
    )
    parser.add_argument("--check", action="store_true",
                        help="verify an existing sidecar and exit; build nothing")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if a valid sidecar is already present")
    parser.add_argument("--store-dir", default=None,
                        help="store whose split lists the sidecar must cover "
                             "(default: <dataroot>/store_det2d)")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    out = Path(args.out) if args.out else dataroot / SIDECAR_NAME
    store_dir = Path(args.store_dir) if args.store_dir else dataroot / "store_det2d"

    status = verify(out, store_dir)
    if args.check:
        print(status.message)
        return 0 if status.ok else 1
    if status.ok and not args.force:
        print(f"{status.message}\nNothing to do (pass --force to rebuild).")
        return 0
    if not status.ok:
        print(f"{status.message}\nBuilding...")

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
