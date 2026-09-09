"""Fetch the authors' released UniDepth-LSS weights on demand.

Two different sets of weights are involved in this repo, and only one of them needs any help:

* **The frozen UniDepthV2 ViT-L backbone** — pulled from the HuggingFace hub automatically by
  `UniDepthV2.from_pretrained(...)` the first time a model is built, and cached under `$HF_HOME`
  (~1.4 GB). Nothing to do.
* **The UniDepth-LSS head weights** (`feat_adapter` + `bev_trans` + `seg`, 16.9 MB) — published as
  a GitHub release asset, not on the hub, so nothing fetches them implicitly. That is what this
  module handles.

The head weights are needed to evaluate, benchmark or visualize the published model; they are
*not* needed to train from scratch.

**The digest is pinned deliberately.** Every number in `docs/baseline.md` was measured against
this exact file, so verifying it is what makes "reproduces the published 0.4940255" a claim about
a specific artifact rather than about whatever currently sits at that URL. A mismatch is treated
as an error, not a warning.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

RELEASE_URL = (
    "https://github.com/adeelmhr/UniDepth-LSS/releases/download/v1.0/UniDepthLSS.pt"
)
RELEASE_SHA256 = "3b539df181881b9ec1f30d0f93d310b9210e83d428113801f40210996b6b5c3f"
RELEASE_BYTES = 16_934_139

#: Where `configs/eval.yaml` and the docs expect it. Relative to the repo root.
DEFAULT_RELATIVE_PATH = Path("checkpoints/UniDepthLSS.pt")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def is_release_checkpoint(path: Path) -> bool:
    """True if `path` is the released checkpoint, verified by digest."""
    return path.is_file() and sha256_of(path) == RELEASE_SHA256


def download_release_checkpoint(
    path: Path, *, force: bool = False, progress: bool = True
) -> Path:
    """Download the released checkpoint to `path`, verify it, and return the path.

    Idempotent: an existing file with the right digest is left alone. Downloads through a
    temporary file in the same directory and renames only after the digest checks out, so an
    interrupted or corrupted transfer can never leave a half-written file that later looks like a
    valid checkpoint.
    """
    path = Path(path)

    if path.is_file() and not force:
        if is_release_checkpoint(path):
            log.info(f"checkpoint already present and verified: {path}")
            return path
        raise FileExistsError(
            f"{path} exists but its sha256 does not match the released v1.0 checkpoint.\n"
            f"This is most likely one of your own training checkpoints saved under the release's "
            f"name. Move it aside, or pass force=True / --force to overwrite it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"downloading {RELEASE_URL}\n     -> {path} ({RELEASE_BYTES / 1e6:.1f} MB)")

    # A `\r` progress bar is right on a terminal and actively harmful in a Slurm log, where every
    # refresh is retained as literal text -- this one produced 55 KB of output for a 17 MB file.
    # So: carriage-return updates only on a tty, and coarse one-line milestones otherwise.
    interactive = progress and sys.stdout.isatty()
    milestones = {"last": -1}

    def hook(count: int, block_size: int, total: int) -> None:
        if not progress or total <= 0:
            return
        done = min(count * block_size, total)
        pct = 100.0 * done / total
        if interactive:
            print(f"\r  {pct:5.1f}%  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        else:
            step = int(pct // 25) * 25
            if step > milestones["last"]:
                milestones["last"] = step
                log.info(f"  {step:3d}%  {done / 1e6:.1f} / {total / 1e6:.1f} MB")

    handle = tempfile.NamedTemporaryFile(
        delete=False, dir=path.parent, prefix=path.name + ".", suffix=".part"
    )
    handle.close()
    temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(RELEASE_URL, temporary, reporthook=hook if progress else None)
        if interactive:
            print()

        actual = sha256_of(temporary)
        if actual != RELEASE_SHA256:
            raise RuntimeError(
                f"downloaded file does not match the pinned digest.\n"
                f"  expected {RELEASE_SHA256}\n"
                f"  got      {actual}\n"
                f"Refusing to install it: every number in docs/baseline.md was measured against "
                f"the expected artifact."
            )
        shutil.move(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()

    log.info(f"verified and installed: {path}")
    return path


def resolve_repo_root(start: Path | None = None) -> Path:
    """Repo root by the same `.project-root` marker `rootutils` uses."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".project-root").is_file():
            return candidate
    return Path.cwd()


def ensure_checkpoint(path: str | os.PathLike | None, *, auto_download: bool = True) -> Path:
    """Resolve a checkpoint path, fetching the released weights if that is what is missing.

    Auto-download applies **only** to the release checkpoint at its canonical location. A missing
    path that points anywhere else is one of the user's own runs, and silently downloading
    somebody else's weights over that request would be far worse than failing: the run would
    complete and report numbers for the wrong model.
    """
    root = resolve_repo_root()
    resolved = Path(path) if path else root / DEFAULT_RELATIVE_PATH
    if not resolved.is_absolute():
        resolved = (root / resolved).resolve()

    if resolved.is_file():
        return resolved

    is_default = resolved == (root / DEFAULT_RELATIVE_PATH).resolve()
    if is_default and auto_download:
        log.info(f"{resolved} is missing -- fetching the released v1.0 checkpoint")
        return download_release_checkpoint(resolved)

    hint = (
        "  uv run tools/download_checkpoints.py\n"
        if is_default
        else "  (this is not the release checkpoint path, so it is not fetched automatically)\n"
    )
    raise FileNotFoundError(f"checkpoint not found: {resolved}\n{hint}")
