# Precision and torch.compile

Measured with `dev/compile_bf16_sweep.py`, 3 repeats, 30 timed iterations after 10 warmup, RTX
5090, 448x798 input, batch 1, the **standard 200x200 window**, released v1.0 head weights. Raw
rows in `dev/outputs/sweep_results.csv` and `dev/outputs/combined_benchmark.csv`; the rendered
report is `dev/outputs/results.html`.

Reproduce:

```bash
uv run dev/compile_bf16_sweep.py checkpoint_path=checkpoints/UniDepthLSS.pt \
    compile_stages=backbone num_warmup_iters=10 num_iters=30 num_repeats=3
```

## What ships

```yaml
trainer.precision: "bf16-mixed"
compile_stages: "backbone"
compile_mode: "default"
```

| configuration | ms/sample | Hz | peak memory |
|---|---|---|---|
| fp32 eager | 281.205 ± 0.312 | 3.6 | 5976.6 MB |
| fp32 + backbone compiled | 266.956 ± 0.198 | 3.7 | 3665.4 MB |
| bf16 eager | 116.872 ± 0.016 | 8.6 | 5663.9 MB |
| **bf16 + backbone compiled** | **94.104 ± 0.047** | **10.6** | **3153.0 MB** |

End to end that is **2.99x faster and 47% less memory** than fp32 eager. The two changes are
roughly independent: bf16 alone is 2.40x, compiling the backbone adds 19.5% on top of it.

## Per-stage sweep

Decisions are made on the **whole-model** speedup, not the per-stage ratio — see "Two
measurement pitfalls" below.

| stage | share of fp32 pass | whole-model speedup (fp32) | whole-model speedup (bf16) | decision |
|---|---|---|---|---|
| `backbone` | not hookable | +4.92% ± 0.20 | **+19.52% ± 0.10** | **enabled** |
| `bev_trans` | 35.7% | −0.01% ± 0.48 | +0.57% ± 0.09 | eager |
| `voxel_pool` | not hookable | −0.27% ± 0.28 | −0.03% ± 0.12 | eager |
| `lift_to_ego` | not hookable | −0.13% ± 0.19 | +0.05% ± 0.13 | eager |
| `seg` | 0.13% | −0.20% ± 0.49 | +0.06% ± 0.04 | eager |
| `feat_adapter` | 0.03% | +0.06% ± 0.20 | +0.19% ± 0.16 | eager |

`backbone` at +19.52% falls in the 10–20% band, so it was put to the user explicitly rather than
enabled by default; it was chosen largely for the 44% memory saving.

Everything except the backbone is either noise or a small regression from compile overhead. Note
`bev_trans` is **35.7% of the fp32 pass** but only **11.5% under bf16** (100.0 ms → 13.5 ms): the
BEV transformer is where fp32 hurts most, which is unsurprising given its sequence length is the
grid *area* (40,000 tokens at 200x200).

## Two measurement pitfalls found here

Both were real, both silently produced wrong answers, and both are worth checking on any repo
with one dominant stage.

**1. A per-stage speedup ratio amplifies jitter by the stage's share of the pass.** The template
derived each stage's compiled latency from the whole-pass wall-clock delta and then divided by
that stage's own eager latency. For `feat_adapter` — 0.03% of the pass — ±1 ms of run-to-run
jitter in a 281 ms pass becomes a ±795% "speedup". The first sweep duly classified
`feat_adapter` as **+406%** and recommended compiling it. Fixed by deciding on the whole-pass
ratio, which is defined for every stage and is the question anyone actually cares about.

**2. "Not observed" is not "unsafe".** The bf16 probe reports three outcomes: `ok`, a real
failure, and `not observed (no tensor output captured)` — meaning no forward hook fired. Treating
the third as unsafe excluded `backbone` from the entire bf16 half of the sweep and printed
`skipped (bf16-unsafe stage(s): ['backbone'])`, which is how a **+19.5% win on 90% of the runtime
went unmeasured**. The backbone is unhookable because `LiftSplatProjector` calls
`backbone.encode_decode(...)` directly, never the module — the same reason
`torch.compile(model.projector.backbone)` is a silent no-op and the stage must be addressed as
`projector.backbone.encode_decode`.

## Accuracy

Verified, not assumed. `tools/eval.py` accepts `compile_stages`, and the paper-window gate was
re-run with the backbone compiled:

| `compile_stages` | Vehicle IoU (no filter) | Vehicle IoU (vis ≥ 2) |
|---|---|---|
| `none` | 0.494021 | 0.528456 |
| `backbone` | 0.494048 | 0.528474 |

Δ = 2.7e-5, from Inductor reordering float operations — smaller than the fp32-vs-fp16 gap to the
published number. See `docs/baseline.md`.

`bf16-mixed`, by contrast, *does* change numerics. It is the shipped training precision, as it is
across this family of repos, but the published-number reproduction in `docs/baseline.md` runs at
`trainer.precision=32-true`, so precision is not a confound in that comparison.
