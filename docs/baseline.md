# Baseline and acceptance gates

All numbers: nuScenes val, 6,019 samples, images 448x798, the authors' released v1.0 weights
(`checkpoints/UniDepthLSS.pt`, epoch 10), batch 1, RTX 5090.

Reproduce with:

```bash
uv run tools/eval.py checkpoint_path=checkpoints/UniDepthLSS.pt \
    data.img_size.h=448 data.img_size.w=798 trainer.precision=32-true
```

## Gate 1 — the port is faithful ✅

The acceptance bar for the adaptation: the restructured code must reproduce the authors' own
number on the authors' own window.

| | Vehicle IoU (no filter) | Vehicle IoU (vis ≥ 2) |
|---|---|---|
| Published (paper Table 1, 448x800 column) | 49.4 | 52.8 |
| Checkpoint's own stored `val_iou` | 0.4940255 | — |
| **This port** (fp32, `efficient_attention=false`) | **0.494021** | **0.528456** |

Agreement to 5 decimal places (Δ = 4.5e-6). The residual is fp32-vs-fp16: the released evaluation
ran under `torch.autocast(float16)`, this one in fp32 to remove precision as a variable.

Supporting evidence that the port changed nothing:

- `dev/capture_forward_reference.py` — the `BevGrid` parameterization, the `lift_to_ego`
  extraction, the `torch_scatter` removal and the `components/` split are **bit-identical** to
  the released `Model/model.py` (sha256 per stage, fixed-seed CPU forward).
- `dev/check_dataset_parity.py` — the restructured dataset was byte-identical to the released
  `dataset_nuscenes.py` over 20 samples (checked before that file was removed), and the
  store-backed path is byte-identical to the devkit path (images, intrinsics, extrinsics,
  segmentation, visibility).

### Cost of `efficient_attention`

Enabling SDPA instead of the fused fastpath is not bit-identical (different reduction order).
Measured effect, same weights, paper window:

| `efficient_attention` | no filter | vis ≥ 2 |
|---|---|---|
| `false` (bit-exact) | 0.494021 | 0.528456 |
| `true` (config default) | 0.494021 | 0.528500 |

4.4e-5 on the filtered column, zero on the unfiltered one — numerically immaterial, and it is
what makes the 200x200 window runnable in fp32 at all (6.28 GB vs 27.48 GB).

### Cost of `compile_stages=backbone`

`torch.compile` is not expected to change results, and this verifies that rather than assuming
it. Same weights, paper window, fp32, `efficient_attention=false`:

| `compile_stages` | no filter | vis ≥ 2 |
|---|---|---|
| `none` | 0.494021 | 0.528456 |
| `backbone` | 0.494048 | 0.528474 |

Δ = 2.7e-5 / 1.8e-5 — Inductor reorders float operations slightly. Both are inside the
fp32-vs-fp16 gap to the published 0.4940255, so the +19.5% throughput and −44% memory from
`docs/efficiency.md` cost no measurable accuracy.

## Gate 2 — the same weights on the comparable window ⚠️

| window | cells | extent | area vs Table 1 | no filter | vis ≥ 2 |
|---|---|---|---|---|---|
| `paper` | 128x128 | 64 x 64 m | 41% | 0.494021 | 0.528500 |
| `standard` | 200x200 | 100 x 100 m | 100% | **0.182368** | **0.192507** |
| delta | | | | **+31.2 pp** | **+33.6 pp** |

### Read this before using the 18.2% number

**This is zero-shot window transfer, not UniDepth-LSS's score at the standard setting.** The
released checkpoint was *trained* on the 64 x 64 m window. Scoring it on 100 x 100 m changes two
things at once:

1. The 32–50 m ring was never in its training target at all, so the model has never been asked
   to predict there and its `seg` head was never penalised for missing it.
2. The BEV transformer's sequence length goes from 16,384 to 40,000 tokens, and its sinusoidal
   position encoding is evaluated over a coordinate range 1.6x larger than any seen in training.
   That is a train/test mismatch in the architecture itself, independent of the label window.

So 18.2% is a **lower bound on a transfer experiment**, and it must not be quoted as "UniDepth-LSS
scores 18.2 on the standard setting". What it does establish, and establishes firmly, is the
thing this repo was adapted to check:

> **The published 49.4 / 52.8 are not measured on the same window as any baseline in Table 1, so
> the comparison in that table is not like-for-like.**

For reference, the weakest baseline quoted in Table 1 (CVT) reports 32.5 / 37.7 on the 100 x 100 m
window, and the strongest (GaussianBEV) 43.9 / 50.3.

### What would settle it

A retrain at `data/bev=standard`, which is the only way to get a number that belongs in the same
table as the baselines:

```bash
uv run tools/train.py data/bev=standard trainer.max_epochs=30
uv run tools/eval.py checkpoint_path=outputs/train/<run>/checkpoints/best.ckpt 'grids=[standard]'
```

**Use `max_epochs=30`, not 10**, even though the released checkpoint is from epoch 10.
`max_epochs` sets `CosineAnnealingLR`'s `T_max`, so it changes the whole learning-rate
trajectory, not just when training stops. The released checkpoint's saved optimizer LR is
`0.00022525`, which matches `cosine(t=10, T_max=30)` to 1e-10; under `T_max=10` the LR at that
point would already be fully annealed to `1e-6`. Training for 10 epochs is therefore *not* a
truncation of the published schedule — it is a different one.

### What the released checkpoint tells us about epoch count

The paper states 30 epochs (§3.6). The checkpoint reports `epoch: 10`, carries 10 entries of
training history, and its final logged `val_iou` is 0.4940 — the published number. Its val IoU was
still climbing at that point (0.4878 → 0.4940).

These are consistent: it is a best-on-validation checkpoint, so a 30-epoch run whose best epoch
was 10 would save exactly this, with the history truncated at the moment of saving. What cannot be
determined from the checkpoint alone is whether the run continued past epoch 10 — a save-best-only
run that stopped at 10 and one that ran to 30 without improving are indistinguishable here. Either
way, no epoch after the 10th beat 0.4940.

Until that has run, the honest statement is "the published comparison is not like-for-like", not
"the method scores X on the standard setting".

| | status |
|---|---|
| Gate 1 — port fidelity | ✅ passed |
| Gate 2 — window effect demonstrated | ✅ measured |
| Accuracy unchanged under `compile_stages=backbone` | ✅ verified |
| Training loop runs clean under the shipped defaults, no NaN | ✅ verified |
| Standard-window retrain | ⬜ not yet run |
