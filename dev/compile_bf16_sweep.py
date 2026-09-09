"""TEMPLATE: automated torch.compile + bf16-mixed precision sweep for a research model.

Lives under the target repo's `dev/` folder by deliberate choice (not `tools/`) -- this is a one-off
tuning tool run when introducing/re-checking compile+precision defaults, not a permanent training
entrypoint, even though (unlike a dependency-light `dev/analyze_run.py`-style tool) it needs the full
model stack to actually run forward passes. See skills/fast-torch/SKILL.md for the workflow this is one
step of, and skills/fast-torch/references/*.md for the reasoning behind every decision this file makes.

NOT a drop-in file. Before using it on a real repo:
  1. Fill in `build_synthetic_batch()` to shape a batch like this repo's actual `model.forward()`
     input (see any existing tools/benchmark.py in the target repo for the exact shapes if one
     exists -- reuse its batch-building logic rather than guessing).
  2. Wire `instantiate_model()` to the target's actual Hydra module config.
  3. Fill in `EXTRA_METHOD_STAGES` with any bound-method stage (not an nn.Module child) this repo's
     forward() calls directly -- named_children() finds everything else automatically. See
     references/stage-discovery.md for real examples (camera_unprojection, gaussian_evolution,
     query_init across the lineage) and why these can't be auto-discovered.
  4. Fill in `CONTAINER_CHILD_STAGES` if any top-level nn.Module child is itself a container
     (nn.Sequential/nn.ModuleList) whose own forward() is bypassed -- the parent's forward() manually
     indexes `self.some_container[i](x)` in a Python loop instead of calling the container as a whole
     (confirmed real in JustDepth's `graph_backbone`, an 8-block GNN stack). named_children() finds
     the container fine, but a hook on the container itself never fires (its __call__ is never
     invoked), and compiling+swapping the container attribute would break the `[i]` indexing
     (OptimizedModule doesn't proxy __getitem__). See references/stage-discovery.md.
  5. Fill in `KNOWN_COMPILE_HOSTILE_STAGES` if you already know which stage(s) wrap a vendored CUDA
     op / sparse-conv backend with no fake-tensor kernel (optional -- an undeclared crash is still
     caught and reported, this just labels it immediately). See references/stage-discovery.md.
  6. If the target repo already has a tools/benchmark.py-style per-stage tool (tinycar-dev and
     gcarpred-dev both do), prefer EXTENDING it with this file's bf16-probe + sweep-driver + HTML
     upgrade rather than replacing proven, hand-tuned tooling -- see SKILL.md step 1.
  7. If the target repo's eval/FLOPs-profiling tooling (e.g. a thop.profile() call) runs against the
     SAME model instance you compile, apply compile strictly after that profiling call, not before --
     a compiled submodule is enumerated twice by model.modules() (once as itself, once via its
     OptimizedModule wrapper's delegation), which breaks hook-registration-per-module tools like thop
     with a "buffer already exists"-style error (confirmed real in JustDepth's tools/eval.py).
  8. Add `num_repeats` (int, default 1 -- 3+ recommended) and `compile_stages` (str, default "none")
     fields to this tool's own Hydra config (its `config_name` target, e.g. `configs/fast_torch.yaml`).
     `num_repeats` repeats both the per-stage sweep and the combined benchmark, reporting mean +/-
     stdev and classifying from the mean -- single-run numbers near a decision threshold are noisy
     (confirmed real on JustDepth: one stage's fp32 speedup measured +24.0%, +13.8%, +19.3% on three
     back-to-back runs on the same idle GPU). `compile_stages` lets the combined benchmark (step 10)
     be pinned to an already-decided/shipped stage set instead of re-deriving it from this run's own
     (possibly different) classification -- see references/decision-thresholds.md.

Usage (inside the container, matching the lineage's uv-first convention):
    uv run dev/compile_bf16_sweep.py
    uv run dev/compile_bf16_sweep.py checkpoint_path=/workspace/checkpoints/best.ckpt
    uv run dev/compile_bf16_sweep.py num_iters=100 compile_mode=default

Design notes:
  - Stage discovery is automatic for nn.Module children (via model.named_children()) plus whatever
    bound-method stages are added to EXTRA_METHOD_STAGES, plus whatever container-child stages are
    added to CONTAINER_CHILD_STAGES -- see references/stage-discovery.md for exactly what's automatic
    and what needs a human to fill in.
  - The bf16-compatibility probe and the per-stage latency timing both use forward hooks rather than
    hand-duplicating the model's forward() control flow into a parallel "forward_with_timing()" --
    every existing lineage tools/benchmark.py does that duplication and carries an explicit "keep in
    sync with forward()" comment as an accepted cost. Hooks avoid that cost for anything that's a
    real nn.Module child, at the price of not being able to auto-discover bound-method stages (item 3
    above) or precisely isolate a crash to one submodule inside a single forward() call (see the
    bf16 probe's docstring).
  - Each stage is compiled and timed in isolation, restoring it to eager (in a `finally:`) before
    moving to the next -- matching the safer, catch-and-continue pattern from powerbev-radar's
    tools/benchmark_compile.py, not tinycar-dev/gcarpred-dev's hard-crash-is-fine-for-one-invocation
    convention. This tool is meant to run the *entire* sweep unattended in one invocation, so a crash
    on one stage must not kill the rest of the grid.
  - CONFIRMED ON A REAL GPU (JustDepth, 2026-07-30, see references/case-studies.md): hooking a module
    that IS the current torch.compile target corrupts its own timing -- Dynamo traces the pre/post
    hook's side effects (a Python list append, a torch.cuda.Event.elapsed_time call) as part of the
    compiled graph, since OptimizedModule's __call__ invokes the wrapped module's __call__ (hooks
    included), not just its bare .forward(). Symptoms ranged from silently wrong numbers (garbage
    negative-tens-of-thousands-percent "speedups") to an outright crash once Dynamo also hit its
    recompile limit. Fixed here by never hooking the stage currently being compiled -- its compiled
    latency is instead derived from the whole-pass wall-clock delta (see _timed_pass/run_sweep) --
    which is now the ONLY measurement path for every stage's compiled number, not just a bound-method
    fallback as an earlier draft of this file assumed.
"""
import csv
import datetime
import logging
import statistics
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Callable, Union

import hydra
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

# --- Fill in for the target repo (see module docstring, and references/stage-discovery.md) -------
# name -> attribute name, or a callable(model) -> attribute name for a config-resolved stage (the
# query_init-style case in gcarpred-dev, where the attribute to compile depends on a config value).
EXTRA_METHOD_STAGES: dict[str, Union[str, Callable[[nn.Module], str]]] = {}
CONTAINER_CHILD_STAGES: dict[str, str] = {}

# --- UniDepth-LSS: stages live on a CHILD module, not on the model -----------------------------
# `UniDepthLSS.named_children()` yields only `projector` and `seg` -- far too coarse to act on, and
# the catalogued "a stage lives on a child module, reached only through the child's bound methods"
# edge case in references/use-cases.md. The real stages are one level down inside
# LiftSplatProjector, and two of them (`lift_to_ego`, `_voxel_pool`) are bound methods rather than
# nn.Module children, so they cannot be auto-discovered and cannot be hooked -- only compiled.
#
# stage name -> (child attribute on the model or None for the model itself, attribute on that owner)
STAGE_OWNER: dict[str, tuple[str | None, str]] = {
    # The frozen ViT-L, and the dominant cost -- but targeted as `encode_decode`, NOT as the
    # `backbone` module. LiftSplatProjector.forward calls `self.backbone.encode_decode(...)`
    # directly, so the module's own `__call__`/`forward` never runs: a forward hook on it never
    # fires (the probe reports "not observed"), and `torch.compile(backbone)` would be a silent
    # no-op that compiles a callable nothing invokes.
    "backbone": ("projector.backbone", "encode_decode"),
    "feat_adapter": ("projector", "feat_adapter"),
    "lift_to_ego": ("projector", "lift_to_ego"),  # bound method
    "voxel_pool": ("projector", "_voxel_pool"),   # bound method
    "bev_trans": ("projector", "bev_trans"),
    "seg": (None, "seg"),
}
# Stages reached through their own `__call__`, and so hookable. Everything else is compiled and
# timed by whole-pass wall-clock delta instead.
HOOKABLE_STAGES = ("feat_adapter", "bev_trans", "seg")


def _resolve_owner(model: nn.Module, path: str | None):
    """Resolve a dotted child path ('projector.backbone') to the object holding the attribute."""
    owner = model
    if path:
        for part in path.split("."):
            owner = getattr(owner, part)
    return owner


def _stage_owner(model: nn.Module, stage_name: str):
    child, attr = STAGE_OWNER[stage_name]
    return _resolve_owner(model, child), attr


KNOWN_COMPILE_HOSTILE_STAGES: tuple[str, ...] = ()
# ---------------------------------------------------------------------------------------------

THRESHOLD_DEFAULT_PCT = 20.0  # >this measured speedup -> enable by default
THRESHOLD_ASK_PCT = 10.0  # >this (and <=THRESHOLD_DEFAULT_PCT) -> ask the user; else -> leave eager


def instantiate_model(cfg: DictConfig) -> nn.Module:
    """Build UniDepthLSS from the same Hydra node tools/train.py uses.

    Real pretrained weights matter here: the frozen ViT-L is the dominant cost and its runtime is
    weight-independent, but the BEV transformer's is not entirely -- and a bf16 probe run on
    randomly-initialised weights can miss an overflow that only real activations produce. Pass
    `checkpoint_path=...` to load the released head weights on top.

    Compiled AFTER any checkpoint load, never before: torch.compile wraps a module in
    OptimizedModule and prefixes its state-dict keys with `_orig_mod.`.
    """
    model = hydra.utils.instantiate(cfg.module.model)
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path:
        from tools.eval import load_checkpoint_into
        from unidepthlss.utils.checkpoints import ensure_checkpoint

        checkpoint_path = str(ensure_checkpoint(checkpoint_path))
        load_checkpoint_into(model, checkpoint_path)
        log.info(f"loaded head weights from {checkpoint_path}")
    else:
        log.warning("no checkpoint_path -- benchmarking randomly-initialised head weights")
    return model


def build_synthetic_batch(cfg: DictConfig, device: str) -> dict:
    """A six-camera batch shaped like the real input.

    UniDepthLSS.forward takes positional tensors, not a dict -- `model(images, intrinsics,
    extrinsics)` -- so `call_model()` below unpacks this. Kept as a dict so it still flows through
    the template's plumbing unchanged.

    Intrinsics are a plausible pinhole and extrinsics a six-camera ring; neither affects latency,
    but degenerate calibration can push the lifted points outside the grid, which would make
    `voxel_pool` measure an unrepresentative near-empty scatter.
    """
    height = int(cfg.data.img_size.h)
    width = int(cfg.data.img_size.w)
    batch_size = int(cfg.trainer.batch_size)
    num_cameras = int(cfg.get("num_cameras", 6))

    generator = torch.Generator().manual_seed(0)
    images = torch.rand(
        batch_size, num_cameras, 3, height, width, generator=generator
    ).to(device)

    intrinsics = torch.zeros(batch_size, num_cameras, 3, 3, device=device)
    intrinsics[..., 0, 0] = float(width)
    intrinsics[..., 1, 1] = float(width)
    intrinsics[..., 0, 2] = width / 2.0
    intrinsics[..., 1, 2] = height / 2.0
    intrinsics[..., 2, 2] = 1.0

    extrinsics = torch.zeros(batch_size, num_cameras, 4, 4, device=device)
    for camera in range(num_cameras):
        yaw = torch.tensor(camera * 2.0 * torch.pi / num_cameras)
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        extrinsics[:, camera, 0, 0] = cos_yaw
        extrinsics[:, camera, 0, 1] = -sin_yaw
        extrinsics[:, camera, 1, 0] = sin_yaw
        extrinsics[:, camera, 1, 1] = cos_yaw
        extrinsics[:, camera, 2, 2] = 1.0
        extrinsics[:, camera, 2, 3] = 1.5
        extrinsics[:, camera, 3, 3] = 1.0

    return {"images": images, "intrinsics": intrinsics, "extrinsics": extrinsics}


def call_model(model: nn.Module, batch: dict):
    """UniDepthLSS.forward is positional, so every timed pass goes through here."""
    return model(batch["images"], batch["intrinsics"], batch["extrinsics"])


def discover_stage_modules(model: nn.Module) -> dict[str, list[nn.Module]]:
    """The hookable stages, resolved through STAGE_OWNER rather than `model.named_children()`.

    Auto-discovery is deliberately not used here: it would return `projector` and `seg`, which
    lumps the frozen ViT-L, the adapter, the lifting math, the voxel pooling and the BEV
    transformer into one opaque stage. Only nn.Module stages appear -- the two bound-method
    stages are compiled and timed by whole-pass delta, never hooked.
    """
    stages: dict[str, list[nn.Module]] = {}
    for name in STAGE_OWNER:
        owner, attr = _stage_owner(model, name)
        target = getattr(owner, attr, None)
        if target is None:
            continue
        if name in HOOKABLE_STAGES and isinstance(target, nn.Module) \
                and not isinstance(target, nn.Identity):
            stages[name] = [target]
        else:
            # A bound method: still swept (torch.compile accepts it, and its compiled latency
            # comes from the whole-pass wall-clock delta), but it carries no hook, so it has no
            # isolated eager number. StageLatencyProbe.summary() skips empty sample lists.
            stages[name] = []
    return stages


def all_stage_names(model: nn.Module) -> list[str]:
    """Every stage worth sweeping, hookable or not."""
    names = []
    for name in STAGE_OWNER:
        owner, attr = _stage_owner(model, name)
        if getattr(owner, attr, None) is not None:
            names.append(name)
    return names


@contextmanager
def compiled_stage(model: nn.Module, stage_name: str, mode: str):
    """Attribute-swap torch.compile, restoring the original eager callable(s) in a `finally:` block
    regardless of success or failure -- so one stage crashing never leaves the model in a partially
    -compiled, inconsistent state for the next stage's measurement. CONTAINER_CHILD_STAGES entries
    compile+reassign each child in place via Sequential/ModuleList.__setitem__ instead of swapping the
    container attribute itself (see module docstring).
    """
    compile_kwargs = {} if mode == "default" else {"mode": mode}
    if stage_name in CONTAINER_CHILD_STAGES:
        attr = CONTAINER_CHILD_STAGES[stage_name]
        container = getattr(model, attr)
        originals = [container[i] for i in range(len(container))]
        try:
            for i, orig in enumerate(originals):
                container[i] = torch.compile(orig, **compile_kwargs)
            yield
        finally:
            for i, orig in enumerate(originals):
                container[i] = orig
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
    else:
        # The owner is the projector child for every stage but `seg`; plain getattr/setattr on the
        # model would raise, or silently create a shadowing attribute nothing calls.
        owner, attr = _stage_owner(model, stage_name)
        original = getattr(owner, attr)
        setattr(owner, attr, torch.compile(original, **compile_kwargs))
        try:
            yield
        finally:
            setattr(owner, attr, original)
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()


@contextmanager
def compiled_combined(model: nn.Module, stage_names: set[str], mode: str):
    """Compiles multiple stages TOGETHER -- the actual production `compile_stages` set from step 8,
    not an isolated single stage like `compiled_stage()` above (used for the per-stage sweep in step
    6, which deliberately never combines stages -- see Principles). This is what step 10's combined
    whole-model benchmark measures against: the real shipped configuration, not an approximation from
    summing isolated per-stage deltas (see `whole_model_benchmark`'s docstring for why that sum is
    unreliable). Only `mode="default"` is confirmed safe once more than one stage is compiled
    together -- see `references/decision-thresholds.md`'s CUDA-graph aliasing crash on
    `reduce-overhead`/`max-autotune`. Implemented as a stack of `compiled_stage()` context managers so
    the CONTAINER_CHILD_STAGES vs. plain-attribute handling isn't duplicated.
    """
    with ExitStack() as stack:
        for name in stage_names:
            stack.enter_context(compiled_stage(model, name, mode))
        yield


class StageLatencyProbe:
    """Forward hooks on every stage's module(s), recording per-stage GPU latency across iterations via
    paired torch.cuda.Event, synced individually (so a compiled stage's async kernels are timed
    correctly, and one stage's timing never leaks into another's). Attached once per timed pass and
    covers every discovered stage simultaneously except whichever is passed via `exclude`.

    A stage backed by more than one module (a CONTAINER_CHILD_STAGES entry) has every child's hook
    write into the same name key -- summed per forward pass in summary(), which is the correct
    aggregate cost of "run this stage once" since each child is invoked exactly once per pass (unlike
    a DT_STEPS-style single-module-called-repeatedly stage -- see references/use-cases.md -- this is n
    distinct child modules, each called once).

    CONFIRMED (not just theorized) to corrupt timing if a hooked module is also the current
    torch.compile target -- see the module docstring's 2026-07-30 entry. `exclude` MUST contain the
    name of whichever stage is currently wrapped in torch.compile, if any; that stage's compiled
    latency is derived from the whole-pass wall-clock delta in run_sweep() instead, which never
    touches the compiled module with any instrumentation.

    Bound-method stages (EXTRA_METHOD_STAGES) are NOT covered here either -- a forward hook needs an
    nn.Module to attach to. Their isolated timing also falls back to the same whole-pass delta.
    """

    def __init__(self, stage_modules: dict[str, list[nn.Module]], exclude: frozenset[str] = frozenset()):
        self.samples_ms: dict[str, list[float]] = defaultdict(list)
        self._pending: dict[str, list[torch.cuda.Event]] = defaultdict(list)
        self._handles = []
        for name, modules in stage_modules.items():
            if name in exclude:
                continue
            for module in modules:
                self._handles.append(module.register_forward_pre_hook(self._pre(name)))
                self._handles.append(module.register_forward_hook(self._post(name)))

    def _pre(self, name: str):
        def _hook(module, args):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._pending[name].append(ev)
        return _hook

    def _post(self, name: str):
        def _hook(module, args, output):
            start = self._pending[name].pop(0)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            self.samples_ms[name].append(start.elapsed_time(end))
        return _hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-iteration total ms for each stage: sum ALL recorded samples (across every child module
        and every iteration) and divide by the known iteration count -- correct uniformly whether the
        stage is backed by one module (samples == num_iters) or a CONTAINER_CHILD_STAGES entry's n
        children (samples == num_iters * n), since every child fires exactly once per forward pass.
        """
        out = {}
        n_iters = getattr(self, "_iters_hint", None)
        for name, samples in self.samples_ms.items():
            if not samples:
                continue
            divisor = n_iters if n_iters else len(samples)
            out[name] = {"mean_ms": sum(samples) / divisor, "std_ms": 0.0}
        return out


def _iter_tensors(obj):
    """Yield every torch.Tensor leaf inside an arbitrarily nested dict/list/tuple output structure.
    Several real stages (e.g. an image encoder) return a dict of named tensors, not a bare tensor or
    a plain tuple -- a hook that only checks isinstance(output, Tensor) silently observes nothing for
    those and would falsely report full coverage. Walk the whole structure instead of assuming a shape.

    Also duck-types a sparse-tensor-library container (e.g. spconv.pytorch.SparseConvTensor, or any
    similarly-shaped wrapper from another sparse/graph library) via a `.features` + `.indices`
    attribute pair, rather than isinstance-checking a specific library so this doesn't require
    importing it. Without this, a stage returning such an object (confirmed real: a Sparse UNet
    decoder whose tail conv is nn.Identity, so the raw SparseConvTensor passes through unchanged)
    would silently report "not observed" from the bf16 probe -- false "no problem found" confidence,
    the same failure shape as the dict-output case above but for a container type that fix doesn't
    cover. See references/use-cases.md.
    """
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)
    elif hasattr(obj, "features") and hasattr(obj, "indices"):
        yield from _iter_tensors(obj.features)


def probe_bf16_compatibility(model: nn.Module, stage_modules: dict[str, list[nn.Module]], batch: dict) -> dict[str, str]:
    """One forward pass under bf16 autocast, checking every stage's hooked output(s) for NaN/Inf.

    Returns {stage_name: "ok" | "nan" | "crashed: <message>"}.

    Two things this deliberately does NOT try to paper over:
    - If the whole-model forward call raises, this can report the exception (which usually names the
      failing op/module in its message) but cannot always mechanically isolate WHICH stage caused it,
      since stages aren't independently callable in isolation from inside a single forward(). Read the
      exception message and cross-reference references/precision-escape-hatches.md in that case -- a
      genuine hand-off point to human/Claude judgment, not a gap to force-automate around.
    - This runs on ONE synthetic batch. A clean result is evidence a stage is bf16-safe on the inputs
      actually exercised, not proof for every input distribution -- the real TF32/cdist bug (see
      references/case-studies.md) needed a specific self-distance geometry at a large spatial extent
      that a plain torch.randn batch may never reproduce. Treat "ok" here as "no problem found yet,"
      and still watch for NaN/Inf in real training logs per the sibling repo-adapter skill's
      verification discipline.

    Bound-method stages (EXTRA_METHOD_STAGES) aren't hookable for output capture either -- inspect
    them manually if the whole-model probe raises or shows NaN downstream of one.
    """
    results: dict[str, str] = {}
    # A stage called more than once per forward (e.g. a per-timestep render call invoked once per
    # DT_STEPS-style loop) must have EVERY call checked, not just the last -- an earlier NaN must not
    # be masked by a clean final call. Track "any bad tensor seen" per stage, not "last tensor seen".
    saw_nan: dict[str, bool] = defaultdict(bool)
    saw_any: dict[str, bool] = defaultdict(bool)
    handles = []

    def _capture(name):
        def _hook(module, args, output):
            for t in _iter_tensors(output):
                saw_any[name] = True
                if torch.isnan(t).any() or torch.isinf(t).any():
                    saw_nan[name] = True
        return _hook

    for name, modules in stage_modules.items():
        for module in modules:
            handles.append(module.register_forward_hook(_capture(name)))

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            call_model(model, batch)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: report and continue, don't crash the sweep
        for name in stage_modules:
            results[name] = f"crashed: {type(e).__name__}: {e}"
        return results
    finally:
        for h in handles:
            h.remove()

    for name in stage_modules:
        if not saw_any[name]:
            results[name] = "not observed (no tensor output captured)"
        elif saw_nan[name]:
            results[name] = "nan"
        else:
            results[name] = "ok"
    return results


def _timed_pass(model: nn.Module, batch: dict, stage_modules: dict[str, list[nn.Module]],
                 autocast_ctx, cfg: DictConfig, exclude: frozenset[str] = frozenset()
                 ) -> tuple[dict[str, dict[str, float]], float, float]:
    """Returns (per-stage hook summary [excluding `exclude`], peak memory MB, whole-pass mean ms).
    The whole-pass timing wraps the entire call_model(model, batch) call with CUDA events at the top level --
    never via a hook on a torch.compile'd module (see StageLatencyProbe's docstring) -- so it stays
    valid even when a stage under `exclude` is currently wrapped in torch.compile.
    """
    probe = StageLatencyProbe(stage_modules, exclude=exclude)
    probe._iters_hint = cfg.num_iters
    torch.cuda.reset_peak_memory_stats(cfg.device)
    with torch.no_grad(), autocast_ctx:
        for _ in range(cfg.num_warmup_iters):
            call_model(model, batch)
        torch.cuda.synchronize(cfg.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(cfg.num_iters):
            call_model(model, batch)
        end.record()
        torch.cuda.synchronize(cfg.device)
        whole_pass_ms = start.elapsed_time(end) / cfg.num_iters
    summary = probe.summary()
    probe.remove()
    peak_mb = torch.cuda.max_memory_allocated(cfg.device) / 1024**2
    return summary, peak_mb, whole_pass_ms


def bf16_blocked(status: str | None) -> bool:
    """Is this probe result an actual bf16 failure, as opposed to an unobservable stage?

    The probe reports three kinds of outcome: "ok", a real failure ("nan" / "crashed: ..."), and
    "not observed (no tensor output captured)" -- which means no forward hook fired, NOT that the
    stage misbehaved under bf16. Those are entirely different claims, and conflating them
    silently excludes from the whole bf16 half of the sweep exactly the stages that are hardest
    to hook: bound methods, and nn.Modules the parent reaches through a named method.

    Concretely here: `projector.backbone` is ~90% of the forward pass and is invoked as
    `backbone.encode_decode(...)`, so it can never be hooked -- yet the whole model demonstrably
    runs under bf16 autocast (116.87 ms/sample vs 280.79 fp32). Treating "not observed" as unsafe
    reported "skipped (bf16-unsafe stage(s): ['backbone'])" for the one configuration most worth
    measuring.

    So: unobservable stages are allowed through, and the real acceptance check for them is the
    whole-model bf16 pass plus the accuracy gate, not a per-stage hook.
    """
    if status is None:
        return True
    return not (status == "ok" or status.startswith("not observed"))


def run_sweep(cfg: DictConfig, model: nn.Module, stage_modules: dict[str, list[nn.Module]],
              bf16_ok: dict[str, str], batch: dict) -> list[dict]:
    """The full precision x compile grid: for each precision in (fp32, bf16-mixed) -- skipping bf16
    for any stage the probe flagged as not "ok" -- run one eager baseline pass (all stages, via
    StageLatencyProbe) plus one isolated-compile pass per stage. Every stage's compiled latency is
    derived from the whole-pass wall-clock delta, not a hook on the compiled module itself (see
    StageLatencyProbe's docstring for why hooking the compile target is unsafe).

    Returns a list of result rows, one per (stage, precision), ready to append to the CSV history and
    feed into the HTML report.
    """
    rows: list[dict] = []

    for precision in ("fp32", "bf16"):
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" else nullcontext()
        )

        log.info(f"[{precision}] eager baseline pass ({cfg.num_iters} iterations)...")
        eager_summary, eager_peak_mb, eager_whole_ms = _timed_pass(model, batch, stage_modules, autocast_ctx, cfg)

        for name in stage_modules:
            if precision == "bf16" and bf16_blocked(bf16_ok.get(name)):
                rows.append({
                    "stage": name, "precision": precision, "eager_ms": eager_summary.get(name, {}).get("mean_ms"),
                    "compiled_ms": None, "speedup_pct": None,
                    "decision": f"skipped (bf16 probe: {bf16_ok.get(name, 'unknown')})",
                    "peak_mem_eager_mb": round(eager_peak_mb, 1), "peak_mem_compiled_mb": None,
                })
                continue

            if name in KNOWN_COMPILE_HOSTILE_STAGES:
                log.info(f"[{precision}] {name}: known compile-hostile stage, expect a crash or graph break.")

            try:
                with compiled_stage(model, name, cfg.compile_mode):
                    # exclude={name}: hooking the module currently wrapped in torch.compile corrupts
                    # its own timing (see StageLatencyProbe docstring) -- its compiled latency is
                    # derived below from the whole-pass wall-clock delta instead.
                    _, compiled_peak_mb, compiled_whole_ms = _timed_pass(
                        model, batch, stage_modules, autocast_ctx, cfg, exclude=frozenset({name})
                    )
                eager_ms = eager_summary.get(name, {}).get("mean_ms")
                compiled_ms = (compiled_whole_ms - eager_whole_ms + eager_ms) if eager_ms is not None else None

                # Two ways to express the same measured quantity -- the whole-pass wall-clock
                # delta -- and which one is meaningful depends on the stage's share of the pass.
                #
                # The per-stage ratio below is delta / stage_eager_ms. That is informative for a
                # stage that dominates the pass, and pure noise amplification for one that does
                # not: a 0.1 ms stage inside a 278 ms pass turns +-1 ms of run-to-run jitter into
                # a +-1000% "speedup" (observed here at +-795% before this was fixed). And a
                # stage with no forward hook -- a bound method, or an nn.Module the parent reaches
                # through a named method -- has no stage_eager_ms at all, so it could only ever be
                # reported "unmeasured", which is exactly what happened to the ViT-L backbone that
                # is ~90% of this model's runtime.
                #
                # So the whole-pass ratio is recorded on every row and is what drives the
                # decision. It is the honest question anyway: "does compiling this stage make the
                # model faster?"
                whole_speedup_pct = (eager_whole_ms - compiled_whole_ms) / eager_whole_ms * 100
                stage_share_pct = (eager_ms / eager_whole_ms * 100) if eager_ms else None
                if eager_ms and compiled_ms is not None:
                    speedup_pct = (eager_ms - compiled_ms) / eager_ms * 100
                else:
                    speedup_pct = None
                decision = classify(whole_speedup_pct)
                rows.append({
                    "stage": name, "precision": precision, "eager_ms": eager_ms, "compiled_ms": compiled_ms,
                    "speedup_pct": round(speedup_pct, 1) if speedup_pct is not None else None,
                    "whole_eager_ms": round(eager_whole_ms, 3),
                    "whole_compiled_ms": round(compiled_whole_ms, 3),
                    "whole_speedup_pct": round(whole_speedup_pct, 2),
                    "stage_share_pct": round(stage_share_pct, 2) if stage_share_pct is not None else None,
                    "decision": decision,
                    "peak_mem_eager_mb": round(eager_peak_mb, 1), "peak_mem_compiled_mb": round(compiled_peak_mb, 1),
                })
            except Exception as e:  # noqa: BLE001 -- a per-stage crash must not kill the rest of the sweep
                log.warning(f"[{precision}] {name}: compile crashed -- {type(e).__name__}: {e}")
                rows.append({
                    "stage": name, "precision": precision, "eager_ms": eager_summary.get(name, {}).get("mean_ms"),
                    "compiled_ms": None, "speedup_pct": None, "decision": f"crashed: {type(e).__name__}: {e}",
                    "peak_mem_eager_mb": round(eager_peak_mb, 1), "peak_mem_compiled_mb": None,
                })
    return rows


def classify(speedup_pct: float) -> str:
    if speedup_pct > THRESHOLD_DEFAULT_PCT:
        return "default"
    if speedup_pct > THRESHOLD_ASK_PCT:
        return "ask"
    if speedup_pct < 0:
        return "regression"
    return "skip"


def whole_model_benchmark(model: nn.Module, batch: dict, autocast_ctx, cfg: DictConfig) -> dict[str, float]:
    """Whole-model latency + peak memory, with NO per-stage hooks attached at all -- the true
    end-to-end number, not an approximation from summing isolated per-stage measurements. Confirmed
    on a real run (JustDepth, 2026-07-30) that the two disagree: per-stage bf16 `eager_ms` values from
    run_sweep() summed to ~9.8ms, while this clean whole-pass measurement came in at 7.1ms -- the
    difference is hook overhead (StageLatencyProbe's per-stage `torch.cuda.synchronize()` calls),
    which the isolated per-stage sweep pays but real inference never does. See
    `references/case-studies.md`.
    """
    torch.cuda.reset_peak_memory_stats(cfg.device)
    with torch.no_grad(), autocast_ctx:
        for _ in range(cfg.num_warmup_iters):
            call_model(model, batch)
        torch.cuda.synchronize(cfg.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(cfg.num_iters):
            call_model(model, batch)
        end.record()
        torch.cuda.synchronize(cfg.device)
    ms_per_sample = start.elapsed_time(end) / cfg.num_iters
    peak_mem_mb = torch.cuda.max_memory_allocated(cfg.device) / 1024**2
    hz = 1000.0 / ms_per_sample if ms_per_sample > 0 else float("inf")
    return {"ms_per_sample": ms_per_sample, "hz": hz, "peak_mem_mb": peak_mem_mb}


def _aggregate(samples: list[dict[str, float]]) -> dict[str, float]:
    """Mean + stdev across repeat whole_model_benchmark() calls. stdev is 0.0 for a single sample
    (statistics.stdev requires >=2) rather than raising -- num_repeats=1 is a valid, if noisier, config.
    """
    ms = [s["ms_per_sample"] for s in samples]
    hz = [s["hz"] for s in samples]
    mem = [s["peak_mem_mb"] for s in samples]
    return {
        "ms_per_sample": statistics.mean(ms), "ms_per_sample_std": statistics.stdev(ms) if len(ms) > 1 else 0.0,
        "hz": statistics.mean(hz), "hz_std": statistics.stdev(hz) if len(hz) > 1 else 0.0,
        "peak_mem_mb": statistics.mean(mem), "peak_mem_mb_std": statistics.stdev(mem) if len(mem) > 1 else 0.0,
    }


def run_combined_benchmark(cfg: DictConfig, model: nn.Module, batch: dict,
                            compile_stage_names: set[str], bf16_ok: dict[str, str],
                            num_repeats: int = 1) -> list[dict]:
    """Step 10: the whole-model, chosen-stages-together comparison -- eager (nothing compiled) vs.
    the actual `compile_stages` set from step 8, compiled as ONE combined configuration
    (`compile_mode="default"` only, per Principles). This is the number that answers "how much faster
    is inference, really" -- distinct from run_sweep()'s per-stage isolated grid, which never compiles
    more than one stage at a time and whose per-stage numbers should not be summed to approximate this
    (see whole_model_benchmark's docstring).

    `num_repeats` repeats the whole_model_benchmark() TIMING pass `num_repeats` times per precision
    (mean +/- stdev reported), WITHOUT recompiling between repeats -- torch.compile is entered once
    per precision and the timed forward-pass loop runs num_repeats times inside that single compiled
    context. This measurement is noticeably more stable run-to-run than run_sweep()'s per-stage delta
    numbers (confirmed on JustDepth: combined fp32 eager latency varied by only ~3% of its mean across
    3 repeats, vs. individual per-stage speedups varying by 10-40 points of their own mean over the
    same 3 repeats) -- see `references/decision-thresholds.md`.

    A precision is skipped entirely if ANY chosen stage failed the bf16 probe -- running the combined
    configuration under bf16 would be unsafe/invalid if even one of its stages isn't bf16-clean,
    regardless of what the OTHER stages' probe results say.
    """
    rows: list[dict] = []
    for precision in ("fp32", "bf16"):
        if precision == "bf16" and any(bf16_blocked(bf16_ok.get(name)) for name in compile_stage_names):
            unsafe = sorted(n for n in compile_stage_names if bf16_blocked(bf16_ok.get(n)))
            log.warning(f"Skipping combined bf16 benchmark -- bf16-unsafe stage(s) in compile_stages: {unsafe}")
            rows.append({
                "precision": "bf16", "eager_ms": None, "eager_ms_std": None, "eager_hz": None, "eager_hz_std": None,
                "eager_mem_mb": None, "compiled_ms": None, "compiled_ms_std": None, "compiled_hz": None,
                "compiled_hz_std": None, "compiled_mem_mb": None,
                "speedup_pct": None, "note": f"skipped (bf16-unsafe stage(s): {unsafe})",
            })
            continue

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" else nullcontext()
        )
        eager = _aggregate([whole_model_benchmark(model, batch, autocast_ctx, cfg) for _ in range(num_repeats)])

        row = {
            "precision": precision,
            "eager_ms": round(eager["ms_per_sample"], 3), "eager_ms_std": round(eager["ms_per_sample_std"], 3),
            "eager_hz": round(eager["hz"], 1), "eager_hz_std": round(eager["hz_std"], 1),
            "eager_mem_mb": round(eager["peak_mem_mb"], 1),
        }
        if compile_stage_names:
            with compiled_combined(model, compile_stage_names, cfg.compile_mode):
                compiled = _aggregate([whole_model_benchmark(model, batch, autocast_ctx, cfg) for _ in range(num_repeats)])
            speedup_pct = (eager["ms_per_sample"] - compiled["ms_per_sample"]) / eager["ms_per_sample"] * 100
            row.update({
                "compiled_ms": round(compiled["ms_per_sample"], 3), "compiled_ms_std": round(compiled["ms_per_sample_std"], 3),
                "compiled_hz": round(compiled["hz"], 1), "compiled_hz_std": round(compiled["hz_std"], 1),
                "compiled_mem_mb": round(compiled["peak_mem_mb"], 1),
                "speedup_pct": round(speedup_pct, 1), "note": None,
            })
        else:
            row.update({
                "compiled_ms": None, "compiled_ms_std": None, "compiled_hz": None, "compiled_hz_std": None,
                "compiled_mem_mb": None,
                "speedup_pct": None, "note": "compile_stages empty -- nothing to compare",
            })
        rows.append(row)
    return rows


def run_sweep_repeated(cfg: DictConfig, model: nn.Module, stage_modules: dict[str, list[nn.Module]],
                        bf16_ok: dict[str, str], batch: dict, num_repeats: int) -> list[dict]:
    """Runs run_sweep() `num_repeats` times end-to-end (each rep recompiles every stage in turn --
    unlike run_combined_benchmark's single-compile-many-timed-passes design, there's no cheaper way to
    repeat an isolated per-stage measurement) and aggregates each (stage, precision) cell's eager_ms/
    compiled_ms/speedup_pct as mean +/- stdev across reps, re-classifying from the MEAN speedup rather
    than trusting any single run.

    Confirmed necessary on a real GPU (JustDepth, 2026-07-30): a stage's fp32 speedup measured +24.0%,
    then +13.8%, then +19.3% on three back-to-back sweeps on the same idle GPU -- enough to flip its
    classification between "default" and "ask" run to run. A single sweep's classification for a stage
    near either threshold should not be trusted in isolation -- see `references/decision-thresholds.md`.
    """
    reps = [run_sweep(cfg, model, stage_modules, bf16_ok, batch) for _ in range(num_repeats)]
    keyed: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rep_rows in reps:
        for r in rep_rows:
            keyed[(r["stage"], r["precision"])].append(r)

    aggregated: list[dict] = []
    for (stage, precision), rs in sorted(keyed.items()):
        eager_vals = [r["eager_ms"] for r in rs if r.get("eager_ms") is not None]
        compiled_vals = [r["compiled_ms"] for r in rs if r.get("compiled_ms") is not None]
        speedup_vals = [r["speedup_pct"] for r in rs if r.get("speedup_pct") is not None]
        mem_eager = [r["peak_mem_eager_mb"] for r in rs if r.get("peak_mem_eager_mb") is not None]
        mem_compiled = [r["peak_mem_compiled_mb"] for r in rs if r.get("peak_mem_compiled_mb") is not None]

        # The whole-pass ratio is what the decision is made on -- it is defined for every stage,
        # hookable or not, and it does not amplify jitter by the stage's share of the pass.
        whole_vals = [r["whole_speedup_pct"] for r in rs if r.get("whole_speedup_pct") is not None]
        share_vals = [r["stage_share_pct"] for r in rs if r.get("stage_share_pct") is not None]

        if whole_vals:
            whole_mean = statistics.mean(whole_vals)
            row = {
                "stage": stage, "precision": precision,
                "eager_ms": round(statistics.mean(eager_vals), 3) if eager_vals else None,
                "compiled_ms": round(statistics.mean(compiled_vals), 3) if compiled_vals else None,
                "speedup_pct": round(statistics.mean(speedup_vals), 1) if speedup_vals else None,
                "speedup_pct_std": (
                    round(statistics.stdev(speedup_vals), 1) if len(speedup_vals) > 1 else 0.0
                ),
                "whole_speedup_pct": round(whole_mean, 2),
                "whole_speedup_pct_std": (
                    round(statistics.stdev(whole_vals), 2) if len(whole_vals) > 1 else 0.0
                ),
                "stage_share_pct": round(statistics.mean(share_vals), 2) if share_vals else None,
                "decision": classify(whole_mean),
            }
        else:
            # every rep skipped/crashed for this cell -- surface the (first) reason as-is
            row = {
                "stage": stage, "precision": precision,
                "eager_ms": round(statistics.mean(eager_vals), 3) if eager_vals else None,
                "compiled_ms": None, "speedup_pct": None, "speedup_pct_std": None,
                "whole_speedup_pct": None, "whole_speedup_pct_std": None, "stage_share_pct": None,
                "decision": rs[0]["decision"],
            }
        row["peak_mem_eager_mb"] = round(statistics.mean(mem_eager), 1) if mem_eager else None
        row["peak_mem_compiled_mb"] = round(statistics.mean(mem_compiled), 1) if mem_compiled else None
        row["num_repeats"] = num_repeats
        aggregated.append(row)
    return aggregated


def append_csv_rows(csv_path: Path, timestamp: str, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp"] + list(rows[0].keys()) if rows else []
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({"timestamp": timestamp, **row})


def read_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_STATUS = {
    "default": ("good", "✅ default"),
    "ask": ("warning", "⚠️ ask"),
    "regression": ("critical", "\U0001f53a regression"),
    "skip": ("muted", "– skip"),
}


def _status_for(row: dict) -> tuple[str, str]:
    decision = row.get("decision") or ""
    if decision.startswith("crashed"):
        return "serious", "\U0001f4a5 crash"
    if decision.startswith("skipped"):
        return "muted", "– bf16 skip"
    return _STATUS.get(decision, ("muted", decision or "?"))


def _stat_tile(label: str, value: str, std: float = None, delta_pct: float = None, up_is_good: bool = True) -> str:
    std_html = f'<div class="stat-range">± {std:.2f}</div>' if std else ""
    delta_html = ""
    if delta_pct is not None:
        good = (delta_pct > 0) == up_is_good
        status = "good" if good else "critical"
        sign = "+" if delta_pct >= 0 else ""
        delta_html = f'<div class="stat-delta status-{status}">{sign}{delta_pct:.1f}% vs eager</div>'
    return f"""<div class="stat-tile"><div class="stat-label">{_esc(label)}</div><div class="stat-value">{_esc(value)}</div>{std_html}{delta_html}</div>"""


def _stat_group(prec_label: str, row: dict) -> str:
    """4 cards for one precision: eager/compiled latency + eager/compiled throughput. Delta color
    follows the dataviz skill's stat-tile contract (direction x whether up is good) -- lower is good
    for latency, higher is good for Hz.
    """
    eager_ms, compiled_ms = row["eager_ms"], row["compiled_ms"]
    eager_hz, compiled_hz = row["eager_hz"], row["compiled_hz"]
    ms_delta_pct = (compiled_ms - eager_ms) / eager_ms * 100
    hz_delta_pct = (compiled_hz - eager_hz) / eager_hz * 100
    return f"""<div class="stat-group">
      <div class="stat-group-label">{_esc(prec_label)}</div>
      <div class="stat-grid">
        {_stat_tile("Eager latency", f"{eager_ms:.2f} ms/sample", row.get("eager_ms_std"))}
        {_stat_tile("Compiled latency", f"{compiled_ms:.2f} ms/sample", row.get("compiled_ms_std"), ms_delta_pct, up_is_good=False)}
        {_stat_tile("Eager throughput", f"{eager_hz:.1f} Hz", row.get("eager_hz_std"))}
        {_stat_tile("Compiled throughput", f"{compiled_hz:.1f} Hz", row.get("compiled_hz_std"), hz_delta_pct, up_is_good=True)}
      </div>
    </div>"""


def render_stat_cards(combined_rows: list[dict]) -> str:
    """The headline numbers a stakeholder actually wants: old vs new latency (ms/sample) and
    throughput (Hz), for EVERY precision that produced a real combined comparison -- fp32-true first
    (the conservative baseline), then bf16-mixed (the more aggressive option), each as its own row of
    4 cards. A precision with no real comparison (compile_stages empty, or bf16 skipped as unsafe) is
    simply omitted rather than shown with placeholder values.
    """
    by_precision = {r["precision"]: r for r in combined_rows}
    groups = [
        _stat_group(label, by_precision[prec])
        for prec, label in (("fp32", "fp32-true"), ("bf16", "bf16-mixed"))
        if by_precision.get(prec) and by_precision[prec].get("compiled_ms") is not None
    ]
    if not groups:
        return ('<div class="card"><p class="sub">No combined benchmark available (compile_stages '
                'empty, or every precision was skipped/unsafe).</p></div>')
    return "\n".join(groups)


def render_html_report(bf16_probe: dict[str, str], rows: list[dict], compile_stages_line: str,
                        combined_rows: list[dict]) -> str:
    """Self-contained HTML report (no external assets), following the dataviz skill's method: a
    two-shade single-hue bar pair per stage (eager = tint, compiled = the full categorical-slot-1
    blue) for the "before -> after" job, and the fixed 4-role status palette (good/warning/serious/
    critical) -- never reused for anything else -- for the auto/ask/crash/regression decision, always
    paired with an icon + label so the color is never the only signal. Leads with `render_stat_cards`'
    headline stat tiles (step 10's combined whole-model comparison), per the dataviz skill's stat-tile
    contract, since that's the number a user actually asked "how much faster is this, really" about.
    """
    max_ms = max((r["eager_ms"] for r in rows if r.get("eager_ms")), default=1.0) or 1.0

    def bar_pair(r: dict) -> str:
        eager_w = 100.0 * (r["eager_ms"] or 0) / max_ms
        compiled_w = 100.0 * (r["compiled_ms"] or 0) / max_ms if r.get("compiled_ms") else 0.0
        status, label = _status_for(r)
        if r.get("speedup_pct") is None:
            speedup = "–"
        elif r.get("speedup_pct_std"):
            speedup = f"{r['speedup_pct']:+.1f}% (±{r['speedup_pct_std']:.1f})"
        else:
            speedup = f"{r['speedup_pct']:+.1f}%"
        return f"""
        <div class="stage-row">
          <div class="stage-name">{_esc(r['stage'])} <span class="badge badge-{status}">{label}</span></div>
          <div class="bar-track"><div class="bar-fill bar-eager" style="width:{eager_w:.1f}%"></div></div>
          <div class="bar-track"><div class="bar-fill bar-compiled" style="width:{compiled_w:.1f}%"></div></div>
          <div class="stage-value">{(r['eager_ms'] or 0):.2f}ms → {(r['compiled_ms'] or 0):.2f}ms <span class="status-{status}">({speedup})</span></div>
        </div>"""

    def _probe_row(name: str, res: str) -> str:
        # Python 3.11 f-strings can't embed a backslash escape in the expression part, so the emoji
        # codepoints are pulled out into plain variables first (also affects render_html_report's
        # other f-strings below if edited -- keep any \U-escaped literal outside the {} braces).
        status = "good" if res == "ok" else ("serious" if res.startswith("crashed") else "critical")
        crash_icon = "\U0001f4a5"
        warn_icon = "\U0001f53a"
        label = "✅ ok" if res == "ok" else (f"{crash_icon} {_esc(res)}" if res.startswith("crashed") else f"{warn_icon} {_esc(res)}")
        return f"""<tr><td>{_esc(name)}</td><td class="status-{status}">{label}</td></tr>"""

    probe_rows = "\n".join(_probe_row(name, res) for name, res in bf16_probe.items())

    by_precision: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_precision[r["precision"]].append(r)

    sections = "\n".join(
        f"""<h2>{'fp32-true' if prec == 'fp32' else 'bf16-mixed'}</h2>
        <div class="card">{"".join(bar_pair(r) for r in sorted(prs, key=lambda r: -(r.get('eager_ms') or 0)))}</div>"""
        for prec, prs in by_precision.items()
    )

    def _combined_row(r: dict) -> str:
        if r.get("note"):
            return f"""<tr><td>{_esc(r['precision'])}</td><td colspan="6" class="status-muted">{_esc(r['note'])}</td></tr>"""
        speedup = f"{r['speedup_pct']:+.1f}%" if r.get("speedup_pct") is not None else "–"
        status = "good" if (r.get("speedup_pct") or 0) > 0 else "critical"
        return f"""<tr>
          <td>{_esc(r['precision'])}</td>
          <td>{r['eager_ms']:.3f}±{r['eager_ms_std']:.3f} ms</td><td>{r['eager_hz']:.1f}±{r['eager_hz_std']:.1f} Hz</td><td>{r['eager_mem_mb']:.1f} MB</td>
          <td>{r['compiled_ms']:.3f}±{r['compiled_ms_std']:.3f} ms</td><td>{r['compiled_hz']:.1f}±{r['compiled_hz_std']:.1f} Hz</td><td>{r['compiled_mem_mb']:.1f} MB</td>
          <td class="status-{status}">{speedup}</td>
        </tr>"""

    combined_table = "\n".join(_combined_row(r) for r in combined_rows)
    stat_cards = render_stat_cards(combined_rows)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>torch.compile / bf16-mixed sweep results</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
    --gridline: #e1e0d9; --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
      --gridline: #2c2c2a; --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #e66767;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --gridline: #2c2c2a; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5;
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #e66767;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    margin: 0; padding: 2rem 1.5rem 4rem;
  }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.05rem; margin: 2.5rem 0 0.75rem; }}
  p.sub {{ color: var(--text-secondary); margin-top: 0; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--gridline); }}
  th {{ color: var(--text-muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
  .stage-row {{ display: grid; grid-template-columns: 220px 1fr 1fr 220px; align-items: center; gap: 0.5rem; padding: 0.4rem 0; }}
  .stage-name {{ font-size: 0.82rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .bar-track {{ background: var(--gridline); border-radius: 4px; height: 10px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; }}
  .bar-eager {{ background: color-mix(in srgb, var(--series-1) 40%, var(--surface-1)); }}
  .bar-compiled {{ background: var(--series-1); }}
  .stage-value {{ font-size: 0.78rem; text-align: right; font-variant-numeric: tabular-nums; }}
  .badge {{ font-size: 0.65rem; padding: 0.05rem 0.4rem; border-radius: 6px; margin-left: 0.35rem; font-weight: 600; }}
  .badge-good {{ background: color-mix(in srgb, var(--good) 20%, transparent); color: var(--good); }}
  .badge-warning {{ background: color-mix(in srgb, var(--warning) 25%, transparent); color: color-mix(in srgb, var(--warning) 70%, black); }}
  .badge-serious {{ background: color-mix(in srgb, var(--serious) 20%, transparent); color: var(--serious); }}
  .badge-critical {{ background: color-mix(in srgb, var(--critical) 20%, transparent); color: var(--critical); }}
  .badge-muted {{ background: var(--gridline); color: var(--text-muted); }}
  .status-good {{ color: var(--good); }} .status-warning {{ color: color-mix(in srgb, var(--warning) 70%, black); }}
  .status-serious {{ color: var(--serious); }} .status-critical {{ color: var(--critical); }} .status-muted {{ color: var(--text-muted); }}
  code {{ background: var(--gridline); padding: 0.15rem 0.4rem; border-radius: 4px; }}
  .stat-group {{ margin-top: 1.25rem; }}
  .stat-group:first-child {{ margin-top: 0.5rem; }}
  .stat-group-label {{ font-size: 0.78rem; color: var(--text-secondary); font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.02em; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }}
  .stat-tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; }}
  .stat-label {{ font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.02em; margin-bottom: 0.4rem; }}
  .stat-value {{ font-size: 1.6rem; font-weight: 600; font-variant-numeric: proportional-nums; }}
  .stat-range {{ font-size: 0.72rem; color: var(--text-muted); margin-top: 0.15rem; }}
  .stat-delta {{ font-size: 0.78rem; margin-top: 0.3rem; font-weight: 600; }}
</style>
</head>
<body data-theme-root>
<div class="wrap">
  <h1>torch.compile / bf16-mixed sweep results</h1>
  <p class="sub">Generated {_esc(datetime.datetime.now().isoformat(timespec='seconds'))}</p>

  {stat_cards}

  <h2>bf16-compatibility probe</h2>
  <div class="card"><table><thead><tr><th>Stage</th><th>Result</th></tr></thead><tbody>{probe_rows}</tbody></table></div>

  {sections}

  <h2>Combined whole-model benchmark (chosen stages compiled together)</h2>
  <p class="sub">Eager (nothing compiled) vs. every stage in compile_stages compiled as one configuration -- not a sum of the isolated per-stage numbers above (see whole_model_benchmark's docstring for why those two disagree).</p>
  <div class="card"><table><thead><tr>
    <th>Precision</th>
    <th>Eager ms/sample</th><th>Eager Hz</th><th>Eager peak mem</th>
    <th>Compiled ms/sample</th><th>Compiled Hz</th><th>Compiled peak mem</th>
    <th>Speedup</th>
  </tr></thead><tbody>{combined_table}</tbody></table></div>

  <h2>Recommended Hydra config</h2>
  <div class="card"><code>compile_stages={_esc(compile_stages_line)}</code></div>
</div>
</body>
</html>"""


@hydra.main(version_base="1.3", config_path="../configs", config_name="fast_torch.yaml")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)  # keep the fp32 and bf16 passes' synthetic batches comparable
    if "cuda" not in cfg.device:
        raise ValueError("This sweep requires a CUDA device.")
    if cfg.compile_mode != "default":
        log.warning(
            "compile_mode != 'default' is confirmed to crash with CUDA-graph aliasing once more than "
            "one stage is compiled together (see references/decision-thresholds.md) -- proceeding "
            "anyway since this sweep compiles one stage at a time, but do not adopt this mode for a "
            "multi-stage production config without re-verifying that combination separately."
        )

    log.info("Instantiating model...")
    model = instantiate_model(cfg).to(cfg.device).eval()
    stage_modules = discover_stage_modules(model)
    log.info(f"Discovered stages: {sorted(stage_modules)}")

    batch = build_synthetic_batch(cfg, cfg.device)

    log.info("Running bf16-compatibility probe...")
    bf16_probe = probe_bf16_compatibility(model, stage_modules, batch)
    for name, result in bf16_probe.items():
        if result != "ok":
            log.warning(f"bf16 probe: {name} -> {result} (see references/precision-escape-hatches.md)")

    # num_repeats: single-run per-stage classifications are noisy near the 20%/10% thresholds
    # (confirmed real -- see references/decision-thresholds.md); repeating and classifying from the
    # MEAN speedup is the fix, not a nicety.
    num_repeats = int(cfg.get("num_repeats", 1) or 1)
    log.info(f"Running the full precision x compile sweep ({num_repeats} repeat(s))...")
    rows = run_sweep_repeated(cfg, model, stage_modules, bf16_probe, batch, num_repeats)

    default_stages = sorted(r["stage"] for r in rows if r["decision"] == "default" and r["precision"] == "fp32")
    ask_stages = sorted(r["stage"] for r in rows if r["decision"] == "ask" and r["precision"] == "fp32")
    compile_stages_line = ",".join(default_stages) if default_stages else "none"

    # Step 10: the actual chosen compile_stages set -- default_stages, or an explicit cfg.compile_stages
    # override (pin the combined benchmark to an already-decided/shipped set instead of whatever THIS
    # run's own sweep classifies, useful once a stage's classification has been seen to flip between
    # runs -- see references/decision-thresholds.md). Extend default_stages with whatever the user
    # chose for any ask-band stage via AskUserQuestion (step 7) before running this, if not using the
    # override. Compiled TOGETHER, measured on the whole model -- not a substitute for run_sweep()
    # above; a separate, later measurement of what actually ships.
    override = str(cfg.get("compile_stages", "none") or "none").strip()
    if override != "none":
        combined_stage_names = {s.strip() for s in override.split(",") if s.strip()}
        log.info(f"Using explicit compile_stages override for combined benchmark: {sorted(combined_stage_names)}")
    else:
        combined_stage_names = set(default_stages)
    log.info(f"Running the combined whole-model benchmark ({num_repeats} repeat(s), chosen stages compiled together)...")
    combined_rows = run_combined_benchmark(cfg, model, batch, combined_stage_names, bf16_probe, num_repeats)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.rule("[bold red]fast-torch sweep results")
    table = Table(title=f"Per-stage decision (fp32, mean of {num_repeats} repeat(s))")
    table.add_column("Stage"); table.add_column("Share of pass")
    table.add_column("Whole-model speedup"); table.add_column("Decision")
    for r in rows:
        if r["precision"] != "fp32":
            continue
        if r.get("whole_speedup_pct") is None:
            whole = "-"
        elif r.get("whole_speedup_pct_std"):
            whole = f"{r['whole_speedup_pct']:+.2f}% (±{r['whole_speedup_pct_std']:.2f})"
        else:
            whole = f"{r['whole_speedup_pct']:+.2f}%"
        share = f"{r['stage_share_pct']:.1f}%" if r.get("stage_share_pct") is not None else "not hookable"
        table.add_row(r["stage"], share, whole, r["decision"])
    console.print(table)
    if ask_stages:
        console.print(f"[yellow]Ask the user about: {ask_stages}[/yellow]")
    console.print(f"Recommended: compile_stages={compile_stages_line!r}")

    combined_table = Table(title=f"Combined whole-model benchmark (mean of {num_repeats} repeat(s), chosen stages compiled together)")
    for col in ("Precision", "Eager ms/sample", "Eager Hz", "Eager mem", "Compiled ms/sample", "Compiled Hz", "Compiled mem", "Speedup"):
        combined_table.add_column(col)
    for r in combined_rows:
        if r.get("note"):
            combined_table.add_row(r["precision"], r["note"], "", "", "", "", "", "")
            continue
        combined_table.add_row(
            r["precision"], f"{r['eager_ms']:.3f}±{r['eager_ms_std']:.3f} ms", f"{r['eager_hz']:.1f}±{r['eager_hz_std']:.1f} Hz",
            f"{r['eager_mem_mb']:.1f} MB",
            f"{r['compiled_ms']:.3f}±{r['compiled_ms_std']:.3f} ms", f"{r['compiled_hz']:.1f}±{r['compiled_hz_std']:.1f} Hz",
            f"{r['compiled_mem_mb']:.1f} MB",
            f"{r['speedup_pct']:+.1f}%",
        )
    console.print(combined_table)

    output_dir = Path(cfg.output_dir)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    if rows:
        append_csv_rows(output_dir / "sweep_results.csv", timestamp, rows)
    if combined_rows:
        append_csv_rows(output_dir / "combined_benchmark.csv", timestamp, combined_rows)
    (output_dir / "results.html").write_text(render_html_report(bf16_probe, rows, compile_stages_line, combined_rows))
    log.info(f"Results appended to <{output_dir / 'sweep_results.csv'}>.")
    log.info(f"Combined benchmark appended to <{output_dir / 'combined_benchmark.csv'}>.")
    log.info(f"HTML report written to <{output_dir / 'results.html'}>.")
    log.info("Sweep completed. If any stage is in the 10-20% band, use AskUserQuestion before wiring "
              "the config -- see SKILL.md workflow step 7.")


if __name__ == "__main__":
    main()
