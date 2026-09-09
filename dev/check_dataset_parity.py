"""Three-way dataset parity: released code vs refactored devkit path vs store path.

    uv run dev/check_dataset_parity.py --dataroot /data/nuscenes --limit 20

Two independent claims get checked here, and both have to hold before the store path can become
the default:

1. **The refactor changed nothing.** Compares the released `dataset_nuscenes.py` directly
   against the restructured `unidepthlss.data.dataset`, covering the shared-rasterizer
   extraction and the `BevGrid` parameterization. This leg **passed 20/20 and the released file
   was then removed**, so it now skips itself. Restore it to re-run:
   `git show 064d3a5:dataset_nuscenes.py > dataset_nuscenes.py`.
2. **The store path is a drop-in.** `NuScenesStoreDataset` must produce identical tensors to the
   devkit path for every field -- images, intrinsics, extrinsics, segmentation and visibility.

Augmentation is off on every leg; it draws from the global `random` stream, so leaving it on
would compare different random draws rather than different code paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from unidepthlss.geometry import BevGrid  # noqa: E402

FIELDS = ("images", "intrinsics", "extrinsics", "segmentation", "visibility")


def compare(name_a, sample_a, name_b, sample_b) -> list[str]:
    problems = []
    for field, a, b in zip(FIELDS, sample_a, sample_b):
        if a.shape != b.shape:
            problems.append(f"{field}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs().max().item()
            problems.append(f"{field}: max|diff| = {diff:g}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", default="/data/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=392)
    args = parser.parse_args()

    img_size = (args.img_height, args.img_width)
    # The released grid, so the legacy dataset (which only knows bev_size/bev_res) is comparable.
    grid = BevGrid(h=128, w=128, h_meters=64.0, w_meters=64.0)
    print(f"grid: {grid.describe()}")
    print(f"img_size: {img_size}\n")

    from unidepthlss.data.dataset import NuScenesBEVDataset
    from unidepthlss.data.store_dataset import NuScenesStoreDataset

    print("building refactored devkit dataset ...")
    devkit = NuScenesBEVDataset(
        dataroot=args.dataroot, version=args.version, split=args.split,
        img_size=img_size, grid=grid, augment=False, return_visibility=True,
    )

    print("building store dataset ...")
    store = NuScenesStoreDataset(
        dataroot=args.dataroot, split=args.split,
        img_size=img_size, grid=grid, augment=False, return_visibility=True,
    )

    legacy = None
    legacy_path = Path(__file__).resolve().parents[1] / "dataset_nuscenes.py"
    if legacy_path.exists():
        print("building legacy released dataset ...")
        sys.path.insert(0, str(legacy_path.parent))
        import dataset_nuscenes  # type: ignore[import-not-found]

        legacy = dataset_nuscenes.NuScenesBEVDataset(
            dataroot=args.dataroot, version=args.version, split=args.split,
            img_size=img_size, bev_size=grid.h, bev_res=grid.res_h,
            augment=False, return_visibility=True,
        )
    else:
        print("legacy dataset_nuscenes.py is gone -- skipping leg 1")

    # Index by token so the three datasets are compared on the SAME samples; the store's split
    # file and the devkit's scene walk do not produce the same ordering.
    devkit_at = {t: i for i, t in enumerate(devkit.sample_tokens)}
    tokens = [t for t in store.sample_tokens if t in devkit_at][: args.limit]
    print(f"\ncomparing {len(tokens)} samples\n")

    legacy_at = {t: i for i, t in enumerate(legacy.sample_tokens)} if legacy else {}

    fail_refactor = fail_store = 0
    for token in tokens:
        devkit_sample = devkit[devkit_at[token]]
        store_sample = store[store.sample_tokens.index(token)]

        if legacy is not None and token in legacy_at:
            problems = compare("legacy", legacy[legacy_at[token]], "devkit", devkit_sample)
            if problems:
                fail_refactor += 1
                print(f"  [refactor] {token}: {'; '.join(problems)}")

        problems = compare("devkit", devkit_sample, "store", store_sample)
        if problems:
            fail_store += 1
            print(f"  [store]    {token}: {'; '.join(problems)}")

    print()
    if legacy is not None:
        status = "✅" if fail_refactor == 0 else "❌"
        print(f"{status} refactor parity : {len(tokens) - fail_refactor}/{len(tokens)} identical "
              f"(released dataset_nuscenes.py vs unidepthlss.data.dataset)")
    status = "✅" if fail_store == 0 else "❌"
    print(f"{status} store parity    : {len(tokens) - fail_store}/{len(tokens)} identical "
          f"(devkit path vs store path)")
    return 0 if (fail_refactor == 0 and fail_store == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
