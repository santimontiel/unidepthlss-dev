"""Inspect a finished run without importing torch or the training stack.

    uv run dev/analyze_run.py                              # the most recent run
    uv run dev/analyze_run.py outputs/train/2026-09-09_14-00-00
    uv run dev/analyze_run.py --all                         # one line per run

Reads only the config snapshot, the CSV metrics and the checkpoint file *sizes* -- deliberately
no `torch.load`, no model construction, nothing that needs the GPU or the environment the run
used. That keeps it fast and immune to dependency drift, which is the whole point of it living
in `dev/` rather than `tools/`.

It also flags the two things most worth knowing about a run in this repo before reading its
numbers at all: which BEV window it used (only `standard` is comparable to the paper's Table 1)
and whether any logged scalar went NaN or Inf.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DEFAULT_ROOT = Path("outputs")


def _load_yaml_ish(path: Path) -> dict:
    """Read the handful of keys we need out of Hydra's config snapshot.

    Uses PyYAML when it is importable (it is, via the dev dependency group) and otherwise
    degrades to returning nothing rather than failing -- this tool must never be the reason
    someone cannot inspect a run.
    """
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def find_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.rglob(".hydra") if p.is_dir()]
    return sorted((p.parent for p in runs), key=lambda p: p.stat().st_mtime)


def read_metrics(run: Path) -> tuple[list[str], list[dict]]:
    candidates = sorted(run.rglob("metrics.csv"))
    if not candidates:
        return [], []
    with candidates[0].open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def numeric(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize(run: Path, verbose: bool = True) -> dict:
    cfg = _load_yaml_ish(run / ".hydra" / "config.yaml")
    data = cfg.get("data", {}) or {}
    bev = data.get("bev", {}) or {}
    module = (cfg.get("module", {}) or {}).get("model", {}) or {}
    trainer = cfg.get("trainer", {}) or {}

    fields, rows = read_metrics(run)

    nan_fields: list[str] = []
    for field in fields:
        for row in rows:
            value = numeric(row.get(field))
            if value is not None and (math.isnan(value) or math.isinf(value)):
                nan_fields.append(field)
                break

    best = {}
    for field in fields:
        if "metrics" not in field:
            continue
        values = [v for v in (numeric(r.get(field)) for r in rows) if v is not None]
        if values:
            best[field] = max(values)

    checkpoints = sorted((run / "checkpoints").glob("*.ckpt")) if (run / "checkpoints").exists() else []

    summary = {
        "run": str(run),
        "bev": f"{bev.get('h', '?')}x{bev.get('w', '?')} @ "
               f"{data.get('depth_max', '?')}m clamp",
        "window": "standard" if bev.get("h") == 200 else
                  ("paper" if bev.get("h") == 128 else "custom"),
        "img_size": f"{(data.get('img_size') or {}).get('h', '?')}x"
                    f"{(data.get('img_size') or {}).get('w', '?')}",
        "source": data.get("source", "?"),
        "precision": trainer.get("precision", "?"),
        "max_epochs": trainer.get("max_epochs", "?"),
        "strict_freeze": module.get("strict_freeze", "?"),
        "efficient_attention": module.get("efficient_attention", "?"),
        "epochs_logged": len({r.get("epoch") for r in rows if r.get("epoch")}),
        "best": best,
        "nan_fields": nan_fields,
        "checkpoints": [c.name for c in checkpoints],
    }

    if not verbose:
        return summary

    print(f"run                 : {summary['run']}")
    print(f"bev window          : {summary['window']}  ({summary['bev']})")
    if summary["window"] == "paper":
        print("                      ^ NOT comparable to the paper's Table 1 baselines")
    print(f"img_size            : {summary['img_size']}")
    print(f"data source         : {summary['source']}")
    print(f"precision           : {summary['precision']}")
    print(f"strict_freeze       : {summary['strict_freeze']}")
    print(f"efficient_attention : {summary['efficient_attention']}")
    print(f"epochs logged       : {summary['epochs_logged']} / {summary['max_epochs']}")
    if best:
        print("best metrics        :")
        for name, value in sorted(best.items()):
            print(f"    {name:34s} {value:.4f}")
    else:
        print("best metrics        : (none logged yet)")
    print(f"checkpoints         : {', '.join(summary['checkpoints']) or '(none)'}")
    if nan_fields:
        print(f"❌ NaN/Inf in       : {', '.join(nan_fields)}")
    elif rows:
        print("✅ no NaN/Inf in any logged scalar")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", default=None)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--all", action="store_true", help="one line per run")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.run:
        runs = [Path(args.run)]
    else:
        runs = find_runs(Path(args.root))
        if not runs:
            print(f"no runs found under {args.root}/")
            return 1
        if not args.all:
            runs = runs[-1:]

    if args.all:
        print(f"{'run':44s} {'window':9s} {'prec':11s} {'best val IoU':>12s}")
        summaries = []
        for run in runs:
            s = summarize(run, verbose=False)
            summaries.append(s)
            iou = s["best"].get("val/metrics/iou")
            print(f"{s['run'][-44:]:44s} {s['window']:9s} {str(s['precision']):11s} "
                  f"{(f'{iou:.4f}' if iou is not None else '-'):>12s}")
        if args.json:
            print(json.dumps(summaries, indent=2))
        return 0

    summaries = [summarize(run) for run in runs]
    if args.json:
        print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
