# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An adaptation of the authors' public code release for *UniDepth-LSS: Depth Foundation Model
Priors for Camera-Only BEV Perception* (DriveX @ ECCV 2026) onto this team's infrastructure.
Camera-only binary vehicle segmentation in BEV on nuScenes: a frozen UniDepthV2 ViT-L supplies
metric depth and image features, and LSS's learned depth-bin distribution is replaced by
depth-guided lifting to a single 3D point per feature cell.

Ported to `uv` + Hydra + PyTorch Lightning + Docker/Slurm, following the `tinycar-dev` shape.

## Commands

Docker-mandatory; everything runs through `uv` inside the container.

```bash
export NUSCENES_DATA_ROOT=/path/to/nuscenes
make build && make run

uv run tools/train.py                             # 200x200 @ 100 m (comparable window)
uv run tools/train.py data/bev=paper              # 128x128 @ 64 m (the release's window)
uv run tools/train.py trainer.devices=2           # Lightning spawns; no torchrun
uv run tools/eval.py checkpoint_path=<ckpt>       # scores BOTH windows in one pass
uv run tools/build_calibration_sidecar.py         # once, before using data.source=store
uv run dev/analyze_run.py                         # post-hoc, no torch import
```

There is **no test suite, by design**. Correctness comes from running the real scripts plus the
`dev/check_*.py` sanity scripts:

```bash
uv run dev/check_instantiate.py            # every config combination instantiates
uv run dev/check_backbone_freeze.py        # strict_freeze behaviour
uv run dev/capture_forward_reference.py --check docs/forward_reference.json
uv run dev/check_store_parity.py           # store vs devkit labels
uv run dev/check_dataset_parity.py         # full sample-level parity
```

## Architecture

`UniDepthLSS.forward(images, intrinsics, extrinsics)` with `images` shaped `(B, 6, 3, H, W)`:

1. **Flatten the rig** to `(B*6, ...)`, ImageNet-normalize.
2. **Frozen backbone** — `UniDepthV2.encode_decode` under `no_grad` returns a metric depth map
   and the last ViT-L encoder features at `H/14 x W/14`.
3. **Align** — `feat_adapter` maps 1024 → 128 channels; depth is resized to the feature grid and
   clamped to `[depth_min, depth_max]`.
4. **`lift_to_ego`** — back-projects a cached pixel-centre frustum at the single predicted depth,
   then maps to the ego frame. This is the paper's substitution for LSS's depth bins.
5. **`_voxel_pool`** — bins points into `voxel_z x h x w`, averages per occupied voxel, then
   **max**-reduces over height to a `(B, 128, h, w)` BEV map.
6. **`bev_trans` + `seg`** — a 2-layer transformer encoder over the flattened BEV tokens with
   sinusoidal position encoding, then a conv head → `(B, 1, h, w)` logits.

Layout follows the reference shape: `modeling/model.py` is pure `nn.Module` wiring,
`modeling/module.py` is the `LightningModule`, interchangeable blocks live in
`modeling/components/`, `tools/` holds Hydra entrypoints, `dev/` holds one-off checks.

## Findings from the port — read before quoting any number

### 1. The evaluation window is not the baselines' window

`configs/data/bev/paper.yaml` is 128 x 128 at 0.5 m = **64 x 64 m (±32 m)**. Every baseline in
the paper's Table 1 reports over 200 x 200 at 0.5 m = **100 x 100 m** (the CVT/LSS "Setting 1"
convention, and what `tinycar-dev/configs/data/nuscenes.yaml` uses). The released window covers
41% of that area and excludes the 32–50 m ring, where camera-only depth is worst — removing
those vehicles from both TP and FN inflates IoU.

Only `data/bev=standard` is comparable to that table. `tools/eval.py` scores both windows from
the same weights; results in `docs/baseline.md`.

**`depth_max` is coupled to the window** and lives in the same config file. The release hardcodes
50.0; a 100 m window needs 61.0 or the outer ring is structurally unreachable and reads as empty.

### 2. The backbone freeze does not hold in the committed code

`unidepth/models/backbones/dinov2.py::DinoVisionTransformer.train()` ends by *unconditionally*
reassigning `requires_grad` on `cls_token`, `pos_embed` and `norm` from its own `frozen_stages`
/ `use_norm` fields, ignoring any external `requires_grad_(False)`. `.eval()` routes through
`train(False)`, so the freeze in `LiftSplatProjector.__init__` is reverted one line after it is
applied, and again on every `.train()` call.

Effect: 1,405,952 backbone parameters stay trainable (`pos_embed` alone is 1,402,880), and the
committed `script/train.ipynb` built its optimizer from `p ... if p.requires_grad` — so running
the released code as committed trains **2,813,633** parameters, not the paper's 1.41M.

**The released checkpoint was not produced that way.** Its optimizer state covers exactly 33
tensors / 1,407,681 elements (feat_adapter + bev_trans + seg) and its `format` field reads
`compact_without_frozen_backbone_v1`. So the paper's figure is right for the real run; this is a
gap between the published *code* and the published *result*.

`module.model.strict_freeze` (config default `true`) re-asserts the freeze after every
`.train()`. Irrelevant to a forward pass, so evaluating a released checkpoint is unaffected.

### 3. Speed: bf16-mixed + a compiled backbone, 2.99x

Shipped defaults are `trainer.precision: bf16-mixed` and `compile_stages: "backbone"`. Measured
(448x798, batch 1, 200x200, RTX 5090): fp32 eager 281.2 ms/sample -> bf16 + compiled backbone
**94.1 ms/sample**, 3.6 -> 10.6 Hz, peak memory 5976.6 -> 3153.0 MB. Accuracy re-verified with
compile on (Δ = 2.7e-5). Full numbers and the two measurement pitfalls behind them:
`docs/efficiency.md`.

`backbone` compiles `projector.backbone.encode_decode`, **not** the `backbone` module —
`LiftSplatProjector` calls that method directly, so compiling the module is a silent no-op and no
forward hook fires on it either. `tools/train.py::_STAGE_OWNER` and
`dev/compile_bf16_sweep.py::STAGE_OWNER` must stay in agreement about this.

### 4. The BEV transformer's fastpath is what makes the comparable window expensive

`bev_trans` attends over *every* BEV cell, so its sequence length is the grid area: 16,384 tokens
at 128x128 but 40,000 at 200x200. PyTorch's fused MultiheadAttention fastpath materializes the
full attention matrix. Measured on an RTX 5090 (448x798, batch 1, 200x200, whole model):

| | fp32 | bf16 |
|---|---|---|
| fastpath on | **27.48 GB** | 5.94 GB |
| fastpath off (SDPA) | **6.28 GB** | 5.94 GB |

`module.model.efficient_attention` (config default `true`) scopes
`torch.backends.mha.set_fastpath_enabled(False)` around the call. It is **not** bit-identical —
SDPA reduces in a different order — so the class default is `false` for exact released numerics.

### 5. The two released notebooks disagree on input resolution

`script/train.ipynb` used `(294, 518)`; `unidepth-lss-evaluate.ipynb` used `(448, 798)` and the
checkpoint directory is `unidepth_lss_img448x798_bev128`. The paper reports 224x476 and 448x798.
The committed training config reproduced neither. `configs/data/nuscenes.yaml` uses 448x798.

## Things that will bite you

**Config carrier keys.** `data.bev`, `data.depth_max` and `data.dataset_name` exist under `data`
so other nodes can interpolate them; Hydra therefore also passes them to `NuScenesDataModule`,
which accepts and ignores them. `data.img_size` is a *mapping* (so `${data.img_size.h}` works),
and `tuple()` on a mapping yields its keys — `_normalize_img_size` handles that. `--cfg job`
does not instantiate, so it will not catch this class of bug; `dev/check_instantiate.py` will.

**`ddp_find_unused_parameters_true` is deliberate**, even though a single-GPU run logs that it
found no unused parameters. Under `strict_freeze=false` the backbone's `pos_embed`/`cls_token`/
`norm` are trainable but sit inside a `torch.no_grad()` block, so they genuinely never receive
gradients — exactly the case plain `"ddp"` errors on. Keep it.

**`num_workers` is coupled to `data.source`.** `store` (default) is fine at 8. `devkit` forks an
~8 GB in-memory index per worker — drop to 2 if you switch.

**bf16 and `.numpy()`.** NumPy has no bfloat16 dtype, so any `.cpu().numpy()` on a model output
raises under `bf16-mixed`. Every such call in `unidepthlss/utils/visualization.py` is preceded by
`.float()`. A forward-only precision probe does not reach this code — only a run that renders a
preview does.

**The BEV grid used to be defined twice.** It is now one `BevGrid`
(`unidepthlss/geometry.py`) read by both the label rasterizer and the voxel pooler. Do not
reintroduce a second definition.

**`BinaryIoU` is dataset-level**, accumulating TP/FP/FN across the split and dividing once — not
a mean of per-batch IoUs. Changing that changes the reported number.

**Metrics are kept per stage, and must stay that way.** Lightning runs validation *inside* the
training epoch, before `on_train_epoch_end`. A single shared metric pair gives: train batches
accumulate -> val batches accumulate into the same counters -> `on_validation_epoch_end` flushes
and resets -> `on_train_epoch_end` flushes empty counters. The visible symptom is
`train/metrics/iou == 1.0000` exactly (BinaryIoU's `union == 0 -> 1.0` fallback); the invisible one
is a `val/metrics/iou` polluted with training batches, which is what `ModelCheckpoint` monitors.
This was a real bug here — fixing it moved a smoke run's val IoU from 0.0259 to 0.0459.

**Load-bearing preserved behaviour.** `LiftSplatProjector.train()` forces `backbone.eval()` on
every call; the Dice term sums over the whole batch rather than per-sample; the optimizer is
AdamW + `CosineAnnealingLR(T_max=max_epochs, eta_min=1e-6)` stepped per epoch, which is the
schedule the paper names explicitly — deliberately not the family's usual OneCycleLR.

**Coordinate convention.** Both the label rasterizer and the voxel pooler use
`row = (half_h - x_ego) / res_h`, `col = (half_w - y_ego) / res_w`. The box draw yaw is
`pi - yaw_ego`: BEV row and column both run opposite to ego x and y, and that double flip is a
reflection, which reverses the sense of rotation. Change one without the other and every box
mirrors.

**`docs/forward_reference.json`** is a bit-exact digest of a fixed-seed CPU forward pass. Re-run
`dev/capture_forward_reference.py --check` after touching anything in `modeling/` — it is the
cheapest proof that a refactor changed nothing, and needs no dataset.
