#!/usr/bin/env python
"""Patchiness curve (goal section 4, secondary): the sharing effect at each uptake level, with intervals. No pass/fail.

    python scripts/v43_curve.py --low runs/curve-low --mid runs/verdict-mid --high runs/curve-high --out curve_out

For S-marks and S-talk against S-solo at each level: the mean and the median paired difference (paired by case and seed) with a
case-clustered 95% bootstrap interval, plus the arm means (S, S-solo, S-marks, S-talk, S'). Each ``--low/--mid/--high`` is a
directory holding ``results.json`` (or the file). Writes ``curve.json`` and ``curve.md`` under ``--out``. Reads only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments.v42_stats import N_RESAMPLES  # noqa: E402
from antelligence.experiments.v43_drive import OutRefused, assert_out_allowed  # noqa: E402
from antelligence.experiments.v43_stats import curve, curve_markdown  # noqa: E402


def _results_path(p: Path) -> Path:
    return p / "results.json" if p.is_dir() else p


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", required=True, type=Path)
    ap.add_argument("--mid", required=True, type=Path)
    ap.add_argument("--high", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: next to the mid results")
    ap.add_argument("--resamples", type=int, default=N_RESAMPLES)
    a = ap.parse_args(argv)
    paths = {"low": _results_path(a.low), "mid": _results_path(a.mid), "high": _results_path(a.high)}
    out = a.out or paths["mid"].parent
    try:
        assert_out_allowed(out)
    except OutRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    by_level: Dict[str, Any] = {k: json.loads(p.read_text()) for k, p in paths.items()}
    for lvl, res in by_level.items():
        got = (res.get("meta") or {}).get("level")
        if got is not None and got != lvl:
            print(f"refused: {paths[lvl]} was run at level {got!r}, not {lvl!r}", file=sys.stderr)
            return 2
    doc = curve(by_level, n=a.resamples)
    doc["sources"] = {k: str(p) for k, p in paths.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "curve.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    md = curve_markdown(doc) + "\nSource: " + ", ".join(f"`{p}`" for p in paths.values()) + ".\n"
    (out / "curve.md").write_text(md)
    print(md)
    print(f"wrote {out / 'curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
