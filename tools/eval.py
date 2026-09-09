"""Evaluate a checkpoint, optionally under several BEV windows in one pass.

    # both windows, same weights -- this is the measurement that matters
    uv run tools/eval.py checkpoint_path=checkpoints/best_model.pt

    # one window only
    uv run tools/eval.py checkpoint_path=<ckpt> 'grids=[paper]'

    # the released setup exactly
    uv run tools/eval.py checkpoint_path=<ckpt> 'grids=[paper]' data.img_size.h=448 \\
        data.img_size.w=798

Reports Vehicle IoU with and without the visibility filter, matching the four settings the
paper's Table 1 quotes.

**Why the multi-grid loop exists.** The released code evaluates over a 128x128 grid at 0.5 m,
i.e. 64 x 64 m. Every baseline in that table reports over 200x200 at 0.5 m, i.e. 100 x 100 m.
The smaller window covers 41% of the area and excludes the 32-50 m ring, where camera-only depth
is worst and vehicles are small, distant and often occluded -- removing them from both the true
positives and the false negatives inflates IoU. None of the model's parameters depend on the
window (the BEV position encoding is computed, not learned), so the *same* state dict loads
under both and the difference is attributable to the window alone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import lightning as L
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

from tools.train import apply_compile, resolve_compile_stages  # noqa: E402
from unidepthlss.metrics import IoUMetric  # noqa: E402
from unidepthlss.utils.checkpoints import ensure_checkpoint  # noqa: E402
from unidepthlss.utils.config import register_new_resolvers, set_seed  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_checkpoint_into(model: torch.nn.Module, checkpoint_path: str) -> dict:
    """Load weights from any of the three checkpoint shapes this repo has to read.

    * the authors' released v1.0 weights -- ``{"model_state_dict": ..., "epoch", "val_iou"}``
    * a Lightning checkpoint from tools/train.py -- ``{"state_dict": {"model.<...>": ...}}``
    * a bare state dict

    `projector.backbone.*` keys are expected to be missing: the frozen UniDepthV2 weights come
    from the HuggingFace hub at construction time and are not carried in the checkpoint. Any
    *other* missing or unexpected key is a real mismatch and raises.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        kind = "released (model_state_dict)"
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {
            key[len("model."):]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
        kind = "lightning (state_dict)"
    else:
        state_dict = checkpoint
        kind = "bare state_dict"

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith("projector.backbone.")]
    unexpected = [k for k in unexpected if not k.startswith("projector.backbone.")]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    info = {"kind": kind, "path": str(checkpoint_path)}
    for key in ("epoch", "val_iou"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            info[key] = checkpoint[key]
    log.info(f"loaded {kind} checkpoint from {checkpoint_path}")
    return info


def grid_override(name: str) -> DictConfig:
    """Load one configs/data/bev/<name>.yaml as a plain node."""
    path = CONFIG_DIR / "data" / "bev" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no BEV window config at {path}")
    return OmegaConf.load(path)


@torch.no_grad()
def evaluate_one(cfg: DictConfig, grid_name: str, device: torch.device) -> dict:
    """Build model + data for one BEV window, load the checkpoint, and score the val split."""
    window = grid_override(grid_name)
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    run_cfg.data.bev = window.bev
    run_cfg.data.depth_max = window.depth_max

    datamodule = hydra.utils.instantiate(run_cfg.data)
    datamodule.setup("validate")
    loader = datamodule.val_dataloader()

    model = hydra.utils.instantiate(run_cfg.module.model)
    # Fetches the released v1.0 weights if -- and only if -- that is the file that is missing.
    # Any other path is one of the user's own runs and must fail rather than be substituted.
    checkpoint_path = ensure_checkpoint(cfg.checkpoint_path)
    info = load_checkpoint_into(model, str(checkpoint_path))
    model = model.to(device).eval()

    # AFTER the checkpoint load, never before: torch.compile wraps a module in OptimizedModule and
    # prefixes its state-dict keys with `_orig_mod.`, which breaks key matching.
    stages = resolve_compile_stages(cfg)
    if stages:
        log.info(f"compiling stage(s) {sorted(stages)} with mode='{cfg.compile_mode}'")
        apply_compile(model, stages, cfg.compile_mode)

    unfiltered = IoUMetric(threshold=cfg.module.iou_threshold, min_visibility=None)
    filtered = IoUMetric(
        threshold=cfg.module.iou_threshold, min_visibility=cfg.task.min_visibility
    )

    use_bf16 = str(cfg.trainer.precision).startswith("bf16")
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    )

    grid = model.grid
    desc = f"{grid_name} ({grid.h}x{grid.w} @ {grid.h_meters:g}m)"
    limit = cfg.get("limit", None)
    total = len(loader) if limit is None else min(int(limit), len(loader))
    seen = 0
    for batch in tqdm(loader, desc=desc, total=total):
        if limit is not None and seen >= int(limit):
            break
        seen += 1
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        extrinsics = batch["extrinsics"].to(device, non_blocking=True)
        target = batch["segmentation"].to(device, non_blocking=True)
        visibility = batch["visibility"].to(device, non_blocking=True)

        with autocast:
            logits = model(images, intrinsics, extrinsics)

        # .float() before the metric: sigmoid on bf16 logits is fine, but keeping the
        # comparison in fp32 removes precision as a variable from a number being compared
        # against a published one.
        logits = logits.float()
        unfiltered.update(logits, target, visibility=visibility)
        filtered.update(logits, target, visibility=visibility)

    return {
        "grid": grid_name,
        "h": grid.h,
        "w": grid.w,
        "h_meters": grid.h_meters,
        "w_meters": grid.w_meters,
        "area_m2": grid.h_meters * grid.w_meters,
        "depth_max": float(model.projector.depth_max),
        "iou": unfiltered.compute(),
        "iou_visible": filtered.compute(),
        "num_samples": seen * int(cfg.trainer.batch_size),
        "num_val_samples": len(datamodule.val_dataset),
        "checkpoint": info,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    register_new_resolvers()
    set_seed(cfg.seed)

    if not cfg.get("checkpoint_path"):
        raise ValueError("checkpoint_path is required, e.g. checkpoint_path=/path/to/best.pt")

    device = torch.device(cfg.trainer.accelerator if torch.cuda.is_available() else "cpu")
    grids = list(cfg.grids) if isinstance(cfg.grids, (list, ListConfig)) else [cfg.grids]

    results = [evaluate_one(cfg, str(name), device) for name in grids]

    print("\n" + "=" * 78)
    print(f"Vehicle IoU  --  {Path(str(cfg.checkpoint_path)).name}")
    print(f"images {cfg.data.img_size.h}x{cfg.data.img_size.w}, "
          f"{results[0]['num_samples']:,} val samples")
    print("=" * 78)
    print(f"{'window':10s} {'cells':>10s} {'extent':>14s} {'no filter':>11s} "
          f"{'vis>=2':>9s}")
    for row in results:
        cells = f"{row['h']}x{row['w']}"
        extent = f"{row['h_meters']:g}x{row['w_meters']:g} m"
        print(
            f"{row['grid']:10s} {cells:>10s} {extent:>14s} "
            f"{row['iou'] * 100:10.2f}% {row['iou_visible'] * 100:8.2f}%"
        )

    if len(results) > 1:
        paper = next((r for r in results if r["grid"] == "paper"), None)
        standard = next((r for r in results if r["grid"] == "standard"), None)
        if paper and standard:
            print("-" * 78)
            print(
                f"{'delta':10s} {'':>10s} {'paper - standard':>14s} "
                f"{(paper['iou'] - standard['iou']) * 100:+10.2f}% "
                f"{(paper['iou_visible'] - standard['iou_visible']) * 100:+8.2f}%"
            )
            ratio = paper["area_m2"] / standard["area_m2"]
            print(f"\nThe 'paper' window covers {ratio:.0%} of the area the Table 1 baselines")
            print("report over. Only the 'standard' row is comparable to that table.")
    print("=" * 78 + "\n")

    # Into this run's own Hydra output dir, not the cwd. `hydra.job.chdir` defaults to False, so
    # writing to Path.cwd() would drop the file in the repo root and let each run overwrite the
    # last one's results.
    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "eval_results.json"
    output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
