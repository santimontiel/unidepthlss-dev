"""Train UniDepth-LSS.

    uv run tools/train.py                                  # 200x200 @ 100 m (comparable)
    uv run tools/train.py data/bev=paper                   # 128x128 @ 64 m (the release)
    uv run tools/train.py module.model.strict_freeze=true  # genuinely freeze the backbone
    uv run tools/train.py logger=csv trainer.devices=2

Lightning owns process spawning, so multi-GPU is `trainer.devices=N` on a single process --
there is no `torchrun` wrapper.
"""

import logging

import hydra
import lightning as L
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

from unidepthlss.utils.config import register_new_resolvers, set_seed  # noqa: E402

# Stages of UniDepthLSS that torch.compile is allowed to target, per the measured sweep in
# dev/compile_bf16_sweep.py. Kept as an explicit allowlist so a typo in `compile_stages` fails
# loudly at startup rather than silently compiling nothing.
COMPILE_SAFE_STAGES = (
    "backbone",
    "feat_adapter",
    "lift_to_ego",
    "voxel_pool",
    "bev_trans",
    "seg",
)

# Stage name -> where it actually lives. Most stages sit on the projector child rather than on
# the model itself, and two are bound methods rather than nn.Module children -- torch.compile
# accepts both.
_STAGE_OWNER = {
    # `encode_decode`, not `backbone`: LiftSplatProjector calls that method directly, so the
    # module's own forward() never runs and compiling the module would silently do nothing.
    "backbone": ("projector.backbone", "encode_decode"),
    "feat_adapter": ("projector", "feat_adapter"),
    "lift_to_ego": ("projector", "lift_to_ego"),
    "voxel_pool": ("projector", "_voxel_pool"),
    "bev_trans": ("projector", "bev_trans"),
    "seg": (None, "seg"),
}


def _resolve_owner(model: torch.nn.Module, path: str | None):
    """Resolve a dotted child path ('projector.backbone') to the object holding the attribute."""
    owner = model
    if path:
        for part in path.split("."):
            owner = getattr(owner, part)
    return owner


def apply_compile(model: torch.nn.Module, stages: set[str], mode: str) -> None:
    """Compile the named stages in place.

    Must run AFTER any checkpoint load: torch.compile wraps a module in OptimizedModule, which
    prefixes its state-dict keys with `_orig_mod.` and breaks key matching.
    """
    compile_kwargs = {} if mode == "default" else {"mode": mode}
    for stage in stages:
        child, attr = _STAGE_OWNER[stage]
        owner = _resolve_owner(model, child)
        setattr(owner, attr, torch.compile(getattr(owner, attr), **compile_kwargs))


def resolve_compile_stages(cfg: DictConfig) -> set[str]:
    stages = {
        stage.strip()
        for stage in str(cfg.get("compile_stages", "none")).split(",")
        if stage.strip() and stage.strip() != "none"
    }
    invalid = stages - set(COMPILE_SAFE_STAGES)
    if invalid:
        raise ValueError(
            f"compile_stages {sorted(invalid)} unknown -- valid stages are "
            f"{list(COMPILE_SAFE_STAGES)}"
        )
    return stages


def instantiate_trainer(cfg: DictConfig) -> L.Trainer:
    loggers = [hydra.utils.instantiate(c) for c in cfg.get("loggers", []) or []]
    callbacks = [hydra.utils.instantiate(c) for c in cfg.get("callbacks", []) or []]
    return L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
    )


def log_training_params(cfg: DictConfig) -> None:
    """Print the resolved knobs that decide how this run trains, before it starts.

    A Slurm log is often the only record of what a run actually did, and the sbatch script can
    only echo what it passed -- not what the config tree resolved those overrides into.
    """
    try:
        choices = HydraConfig.get().runtime.choices
        task_name = choices["task"].removesuffix(".yaml")
        data_name = choices["data"].removesuffix(".yaml")
        bev_name = choices["data/bev"].removesuffix(".yaml")
    except Exception:
        task_name, data_name, bev_name = "<task>", "<data>", "<bev>"

    def sel(key, default="?"):
        return OmegaConf.select(cfg, key, default=default)

    comparable = "comparable to the paper's Table 1" if bev_name == "standard" else \
        "NOT comparable to the paper's Table 1 baselines"

    log.info(
        "\n"
        + "🚀 UniDepth-LSS training run\n"
        + f"   🏷️  run_id            : {sel('run_id')}\n"
        + f"   🧩 task / data        : {task_name} / {data_name}\n"
        + f"   🗺️  bev window        : {bev_name} -- "
          f"{sel('data.bev.h')}x{sel('data.bev.w')} over "
          f"{sel('data.bev.h_meters')}x{sel('data.bev.w_meters')} m ({comparable})\n"
        + f"   📏 depth clamp        : [{sel('module.model.depth_min')}, "
          f"{sel('module.model.depth_max')}] m\n"
        + f"   🖼️  img_size          : {sel('data.img_size.h')}x{sel('data.img_size.w')}\n"
        + f"   🗃️  data source       : {sel('data.source')}\n"
        + f"   🧊 strict_freeze      : {sel('module.model.strict_freeze')}\n"
        + f"   🌱 seed               : {sel('seed')}\n"
        + f"   📦 batch_size         : {sel('trainer.batch_size')} "
          f"(devices {sel('trainer.devices')})\n"
        + f"   👷 num_workers        : {sel('data.num_workers')}\n"
        + f"   🔁 max_epochs         : {sel('trainer.max_epochs')}\n"
        + f"   📈 lr                 : {sel('module.lr')} (cosine -> {sel('module.eta_min')})\n"
        + f"   🎯 precision          : {sel('trainer.precision')}\n"
        + f"   🏆 monitors           : val/metrics/{sel('task.checkpoint_metric')}\n"
        + f"   ⚙️  compile           : {sel('compile_stages') or 'none'} "
          f"(mode={sel('compile_mode')})"
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    register_new_resolvers()
    set_seed(cfg.seed)
    log_training_params(cfg)

    log.info(f"Loading datamodule <{cfg.data._target_}>...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Loading module <{cfg.module._target_}>...")
    module: L.LightningModule = hydra.utils.instantiate(cfg.module, cfg=cfg)

    stages = resolve_compile_stages(cfg)
    if stages:
        log.info(f"Compiling stage(s) {sorted(stages)} with mode='{cfg.compile_mode}'...")
        apply_compile(module.model, stages, cfg.compile_mode)

    trainer = instantiate_trainer(cfg)
    log.info("Starting training...")
    trainer.fit(module, datamodule, ckpt_path=cfg.get("resume_checkpoint", None))
    log.info("Training completed. ✅")


if __name__ == "__main__":
    main()
