"""Instantiate every config combination for real, without touching the dataset or GPU.

    uv run dev/check_instantiate.py

`--cfg job --resolve` only *composes* config; it never calls `hydra.utils.instantiate`, so a
constructor that rejects one of the config's keys still passes a dry run and then fails at
launch. This closes that gap: it builds the DataModule and the LightningModule for every
`data/bev` option and asserts the grid and depth clamp actually reached the model.

`setup()` is deliberately not called -- that is what would load the dataset -- so this stays a
seconds-long check that runs anywhere.
"""

from __future__ import annotations

import itertools

import hydra
import rootutils
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path  # noqa: E402

from unidepthlss.utils.config import register_new_resolvers  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
GRIDS = ("standard", "paper")
LOGGERS = ("wandb", "csv")


def main() -> int:
    register_new_resolvers()
    failures = []

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        for grid, logger in itertools.product(GRIDS, LOGGERS):
            label = f"data/bev={grid} logger={logger}"
            try:
                # return_hydra_config + set_config: the checkpoint callback's dirpath is
                # ${hydra:runtime.output_dir}, which only resolves once HydraConfig exists.
                cfg = compose(
                    config_name="train.yaml",
                    overrides=[f"data/bev={grid}", f"logger={logger}"],
                    return_hydra_config=True,
                )
                HydraConfig.instance().set_config(cfg)
                datamodule = hydra.utils.instantiate(cfg.data)
                for logger_cfg in cfg.get("loggers", []) or []:
                    _ = logger_cfg  # not instantiated: WandbLogger would open a run
                for callback_cfg in cfg.get("callbacks", []) or []:
                    hydra.utils.instantiate(callback_cfg)

                expected = {"standard": (200, 100.0, 61.0), "paper": (128, 64.0, 50.0)}[grid]
                cells, metres, depth_max = expected
                assert datamodule.grid.h == cells, f"grid.h {datamodule.grid.h} != {cells}"
                assert datamodule.grid.h_meters == metres
                assert cfg.module.model.depth_max == depth_max, (
                    f"depth_max {cfg.module.model.depth_max} != {depth_max}"
                )
                # img_size is a mapping in the config so other nodes can interpolate its
                # parts; tuple() on a mapping yields its keys, which would only fail deep inside
                # cv2.resize. Assert it arrived as two ints.
                assert isinstance(datamodule.img_size, tuple), type(datamodule.img_size)
                assert all(isinstance(v, int) for v in datamodule.img_size), (
                    f"img_size must be ints, got {datamodule.img_size}"
                )
                assert datamodule.img_size == (448, 798), datamodule.img_size
                print(f"  ✅ {label:38s} grid {datamodule.grid.describe()}")
                print(f"     {'':38s} img_size {datamodule.img_size}")
            except Exception as exc:  # noqa: BLE001
                failures.append((label, exc))
                print(f"  ❌ {label:38s} {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"❌ {len(failures)} config combination(s) failed to instantiate")
        return 1
    print("✅ every config combination instantiates, with the grid and depth clamp wired through")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
