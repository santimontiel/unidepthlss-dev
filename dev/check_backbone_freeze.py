"""Assert what `strict_freeze` does, and document the upstream behaviour it works around.

    uv run dev/check_backbone_freeze.py

`unidepth/models/backbones/dinov2.py::DinoVisionTransformer.train()` ends by *unconditionally*
reassigning `requires_grad` on `cls_token`, `pos_embed` and `norm` from its own
`frozen_stages`/`use_norm` fields. It ignores any external `requires_grad_(False)`, and `.eval()`
routes through `train(False)` -- so the freeze applied in `LiftSplatProjector.__init__` is
reverted one line after it is applied, and again on every `.train()` call Lightning makes at the
start of an epoch.

Consequences, both worth knowing:

* Running the released code as committed trains 2,813,633 parameters, not the 1.41M the paper
  reports -- the extra 1,405,952 are backbone tensors (`pos_embed` alone is 1,402,880), and the
  committed `script/train.ipynb` builds its optimizer from
  `p for p in model.parameters() if p.requires_grad`.
* The released *checkpoint* was not produced that way: its optimizer state covers exactly 33
  tensors / 1,407,681 elements, and its `format` field reads
  `compact_without_frozen_backbone_v1`. The paper's figure is right for the real run; the
  published code just does not reproduce it.

This script fails loudly if either behaviour changes -- e.g. if a future UniDepth release fixes
`train()`, at which point `strict_freeze` becomes redundant rather than load-bearing.
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from unidepthlss.geometry import BevGrid  # noqa: E402
from unidepthlss.modeling.model import UniDepthLSS  # noqa: E402

#: feat_adapter (147,712) + bev_trans (1,186,048) + seg (73,921).
INTENDED_TRAINABLE = 1_407_681
#: pixel_encoder.{cls_token, pos_embed, norm.weight, norm.bias}
LEAKED_BACKBONE = 1_405_952


def trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(strict_freeze: bool):
    return UniDepthLSS(
        img_height=154,
        img_width=266,
        grid=BevGrid(),
        strict_freeze=strict_freeze,
    )

def main() -> int:
    problems: list[str] = []

    faithful = build(strict_freeze=False)
    strict = build(strict_freeze=True)

    print(f"{'':34s} {'after __init__':>15s} {'after .train()':>15s} {'after .eval()':>14s}")
    for label, model in (("strict_freeze=False", faithful), ("strict_freeze=True", strict)):
        counts = [trainable(model)]
        model.train()
        counts.append(trainable(model))
        model.eval()
        counts.append(trainable(model))
        print(f"{label:34s} {counts[0]:15,d} {counts[1]:15,d} {counts[2]:14,d}")
        if len(set(counts)) != 1:
            problems.append(f"{label}: count changed across train/eval -> {counts}")

    print()
    print(f"intended trainable set  : {INTENDED_TRAINABLE:,}")
    print(f"leaked backbone tensors : {LEAKED_BACKBONE:,}")
    print()

    if trainable(strict) != INTENDED_TRAINABLE:
        problems.append(
            f"strict_freeze=True trains {trainable(strict):,}, expected {INTENDED_TRAINABLE:,}"
        )
    if trainable(faithful) != INTENDED_TRAINABLE + LEAKED_BACKBONE:
        problems.append(
            f"strict_freeze=False trains {trainable(faithful):,}, expected "
            f"{INTENDED_TRAINABLE + LEAKED_BACKBONE:,} -- upstream UniDepth may have fixed "
            f"DinoVisionTransformer.train(), which would make strict_freeze redundant"
        )

    leaked = [
        name
        for name, parameter in faithful.projector.backbone.named_parameters()
        if parameter.requires_grad
    ]
    print("backbone tensors left trainable by the upstream train() override:")
    for name in leaked:
        print(f"   {name}")

    # The other load-bearing contract: the backbone must be in eval mode regardless.
    faithful.train()
    if faithful.projector.backbone.training:
        problems.append("backbone is in training mode after model.train()")

    print()
    if problems:
        for problem in problems:
            print(f"❌ {problem}")
        return 1
    print("✅ strict_freeze behaves as documented, and the backbone stays in eval mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
