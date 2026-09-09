"""Fetch the pretrained weights this repo needs.

    uv run tools/download_checkpoints.py            # the released UniDepth-LSS head weights
    uv run tools/download_checkpoints.py --all      # ...and warm the UniDepthV2 backbone cache
    uv run tools/download_checkpoints.py --force    # re-download even if a file is already there
    uv run tools/download_checkpoints.py --check    # verify only, download nothing

Two sets of weights are involved and they behave differently:

* **UniDepthV2 ViT-L backbone** (~1.4 GB) is pulled from the HuggingFace hub automatically the
  first time a model is constructed, and cached under `$HF_HOME`. You never need this tool for
  it -- `--all` only pre-warms that cache so the first training run does not stall on a download,
  which is worth doing before submitting a cluster job.
* **UniDepth-LSS head weights** (16.9 MB) are a GitHub release asset, so nothing fetches them
  implicitly. This is the one the tool exists for. They are needed to evaluate, benchmark or
  visualize the published model, and not needed to train from scratch.

The head-weights digest is pinned: every number in docs/baseline.md was measured against that
exact file, and a mismatch is a hard error rather than a warning.
"""

from __future__ import annotations

import argparse
import logging

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from unidepthlss.utils.checkpoints import (  # noqa: E402
    DEFAULT_RELATIVE_PATH,
    RELEASE_BYTES,
    RELEASE_SHA256,
    download_release_checkpoint,
    is_release_checkpoint,
    resolve_repo_root,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def warm_backbone_cache() -> None:
    """Construct the backbone once so HuggingFace caches it, then throw it away."""
    log.info("warming the UniDepthV2 backbone cache (~1.4 GB on a cold cache)...")
    from unidepth.models.unidepthv2 import UniDepthV2

    UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    log.info("backbone cached under $HF_HOME")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="also pre-warm the UniDepthV2 backbone cache")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a file is already present")
    parser.add_argument("--check", action="store_true",
                        help="verify what is present; download nothing")
    args = parser.parse_args()

    target = resolve_repo_root() / DEFAULT_RELATIVE_PATH

    if args.check:
        if not target.is_file():
            log.info(f"❌ missing   {target}")
            return 1
        if is_release_checkpoint(target):
            log.info(f"✅ verified  {target}  ({RELEASE_BYTES / 1e6:.1f} MB)")
            return 0
        log.info(f"❌ digest mismatch at {target}\n   expected sha256 {RELEASE_SHA256}")
        return 1

    download_release_checkpoint(target, force=args.force)
    if args.all:
        warm_backbone_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
