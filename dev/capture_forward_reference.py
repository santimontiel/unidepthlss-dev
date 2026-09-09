"""Capture (or re-check) a bit-exact digest of the model's forward pass.

This exists to make a restructuring pass *provable* rather than merely plausible. It runs a
fixed-seed synthetic batch through the model and records a sha256 of every stage's output plus
the final logits. Capture it against the code as it stands, restructure, then re-run with
``--check``: a byte-identical report is proof the move changed nothing, and it needs no dataset,
so it works long before the one-epoch acceptance gate is reachable.

    # the standing check -- the package layout against the captured reference
    uv run dev/capture_forward_reference.py --layout package --check docs/forward_reference.json

    # how the reference was originally captured, before anything moved (the pre-port flat
    # layout has since been removed, so this now needs it restored from git first)
    uv run dev/capture_forward_reference.py --layout original --out docs/forward_reference.json

**CPU and float32, deliberately.** ``_voxel_pool`` accumulates with ``index_add_``, whose CUDA
implementation uses atomics: the summation order varies run to run, so the low bits of the pooled
features -- and therefore every digest downstream of them -- would differ between two identical
GPU runs. On CPU the accumulation is deterministic and the digests are stable. That makes this a
correctness check, not a performance one; it is intentionally not measuring speed.

The image size is small (a few ViT patches) purely so a CPU forward through ViT-L finishes in
seconds. Fidelity here is about *invariance across the refactor*, not about matching the
published resolution -- the one-epoch/eval gates cover that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch  # noqa: E402


# Multiples of 14 (the ViT-L/14 patch size), kept tiny so a CPU forward is quick.
IMG_HEIGHT = 154
IMG_WIDTH = 266
BATCH_SIZE = 1
NUM_CAMERAS = 6
SEED = 42


def _load_model_class(layout: str):
    """Import UniDepthLSS from either the pre- or post-restructure location.

    Kept explicit rather than a try/except chain so a run says which layout it actually
    exercised -- a silent fallback to the wrong one would make a "matching" report meaningless.
    """
    if layout == "original":
        # The released flat layout was removed once the digest below had been captured from it
        # and Gate 1 had passed (see docs/baseline.md). The stored report is the reference now;
        # this path only exists to re-derive it from the pre-port code if that is ever needed.
        original_dir = Path(__file__).resolve().parents[1] / "Model"
        if not (original_dir / "model.py").exists():
            raise SystemExit(
                "--layout original needs the pre-port flat layout, which was removed after Gate 1.\n"
                "docs/forward_reference.json IS the captured reference -- use --layout package\n"
                "--check against it. To re-derive it from the original code:\n"
                "    git show 064d3a5:Model/model.py > Model/model.py"
            )
        sys.path.insert(0, str(original_dir))
        from model import UniDepthLSS  # type: ignore[import-not-found]

        return UniDepthLSS
    from unidepthlss.modeling.model import UniDepthLSS

    return UniDepthLSS


def _digest(tensor: torch.Tensor) -> str:
    """sha256 over a tensor's exact bytes, in a canonical (contiguous, cpu, fp32) form."""
    array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _synthetic_batch() -> dict[str, torch.Tensor]:
    """A fixed, reproducible six-camera batch with plausible calibration."""
    generator = torch.Generator().manual_seed(SEED)
    images = torch.rand(
        BATCH_SIZE, NUM_CAMERAS, 3, IMG_HEIGHT, IMG_WIDTH, generator=generator
    )

    # A plausible pinhole: focal ~ the image width, principal point at the centre.
    intrinsics = torch.zeros(BATCH_SIZE, NUM_CAMERAS, 3, 3)
    intrinsics[..., 0, 0] = float(IMG_WIDTH)
    intrinsics[..., 1, 1] = float(IMG_WIDTH)
    intrinsics[..., 0, 2] = IMG_WIDTH / 2.0
    intrinsics[..., 1, 2] = IMG_HEIGHT / 2.0
    intrinsics[..., 2, 2] = 1.0

    # Six cameras on a ring, yawed 60 degrees apart, 1.5 m above the ego origin.
    extrinsics = torch.zeros(BATCH_SIZE, NUM_CAMERAS, 4, 4)
    for camera in range(NUM_CAMERAS):
        yaw = torch.tensor(camera * 2.0 * torch.pi / NUM_CAMERAS)
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        extrinsics[:, camera, 0, 0] = cos_yaw
        extrinsics[:, camera, 0, 1] = -sin_yaw
        extrinsics[:, camera, 1, 0] = sin_yaw
        extrinsics[:, camera, 1, 1] = cos_yaw
        extrinsics[:, camera, 2, 2] = 1.0
        extrinsics[:, camera, 2, 3] = 1.5
        extrinsics[:, camera, 3, 3] = 1.0

    return {"images": images, "intrinsics": intrinsics, "extrinsics": extrinsics}


def _first_tensor(value):
    """Pull one tensor out of whatever a stage returned (tensor, dict, or sequence)."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def build_report(layout: str) -> dict:
    torch.manual_seed(SEED)
    model_class = _load_model_class(layout)

    model = model_class(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_classes=1,
        feature_channels=128,
    )
    model.initialize_head_bias()
    model.eval()

    # Digest every stage that carries real signal. The frozen backbone is excluded from the
    # per-stage hooks (its output shape depends on UniDepth internals we do not own) but is
    # covered transitively: every digest below sits downstream of it, so a backbone change
    # still shows up.
    stages = {
        "projector.feat_adapter": model.projector.feat_adapter,
        "projector.bev_trans": getattr(model.projector, "bev_trans", None),
        "seg": model.seg,
    }

    digests: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is None:
                return
            digests[name] = _digest(tensor)
            shapes[name] = list(tensor.shape)

        return hook

    for name, module in stages.items():
        if module is not None:
            handles.append(module.register_forward_hook(make_hook(name)))

    batch = _synthetic_batch()
    with torch.no_grad():
        logits = model(batch["images"], batch["intrinsics"], batch["extrinsics"])

    for handle in handles:
        handle.remove()

    digests["output"] = _digest(logits)
    shapes["output"] = list(logits.shape)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {
        "seed": SEED,
        "device": "cpu",
        "dtype": "float32",
        "img_size": [IMG_HEIGHT, IMG_WIDTH],
        "batch_size": BATCH_SIZE,
        "num_cameras": NUM_CAMERAS,
        "param_count_total": total,
        "param_count_trainable": trainable,
        "shapes": shapes,
        "digests": digests,
        # A load-bearing contract, asserted rather than eyeballed: LiftSplatProjector.train()
        # forces the backbone back into eval mode on every call. Lightning calls .train() at the
        # start of every epoch, so losing this silently un-freezes the backbone's norm layers.
        "backbone_eval_after_train_call": _backbone_stays_eval(model),
    }


def _backbone_stays_eval(model) -> bool:
    model.train()
    stayed = not model.projector.backbone.training
    model.eval()
    return stayed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        choices=["original", "package"],
        default="package",
        help="which code layout to import the model from",
    )
    parser.add_argument("--out", type=Path, help="write the report to this path")
    parser.add_argument(
        "--check", type=Path, help="compare against a previously captured report"
    )
    args = parser.parse_args()

    report = build_report(args.layout)

    print(f"layout                : {args.layout}")
    print(f"params total/trainable: {report['param_count_total']:,} / "
          f"{report['param_count_trainable']:,}")
    print(f"backbone stays eval   : {report['backbone_eval_after_train_call']}")
    for name, digest in report["digests"].items():
        print(f"  {name:28s} {tuple(report['shapes'][name])!s:24s} {digest[:16]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")

    if args.check:
        reference = json.loads(args.check.read_text())
        mismatches = [
            (name, reference["digests"].get(name), digest)
            for name, digest in report["digests"].items()
            if reference["digests"].get(name) != digest
        ]
        missing = set(reference["digests"]) - set(report["digests"])
        if not mismatches and not missing:
            print(f"\n✅ forward pass is bit-identical to {args.check}")
            return 0
        print(f"\n❌ forward pass DIVERGED from {args.check}")
        for name, expected, actual in mismatches:
            print(f"   {name}: expected {expected} got {actual}")
        for name in sorted(missing):
            print(f"   {name}: missing from this run")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
