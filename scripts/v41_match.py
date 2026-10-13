#!/usr/bin/env python
"""Raw and S'-resolved match rates per model per question for calibration / step-9 run dirs (read-only).

  .venv/bin/python scripts/v41_match.py RUN_DIR [RUN_DIR ...] [--bonsai] [--json OUT]

"S" means "keep the current target", so a letter only matches S' once both are resolved to the region they refer to
(decision 2026-10-13-conformity-label-sprime-resolved, rule a). Prints both versions. Nothing is written unless --json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v41_fit  # noqa: E402
from antelligence.experiments.v41_resolve import format_match, match_to_sprime  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run output dirs (or dirs containing runs/*/decisions.jsonl)")
    ap.add_argument("--bonsai", action="store_true")
    ap.add_argument("--split-file", type=Path, default=REPO / "config/v4_1/split.json")
    ap.add_argument("--json", type=Path, help="also write the report here")
    a = ap.parse_args(argv)
    try:
        split_doc = v41_fit.load_split(a.split_file) if a.split_file.is_file() else None
        runs = [v41_fit.load_run(d, split_doc) for d in v41_fit.discover(a.runs)]
        samples, skipped = v41_fit.samples_from(runs, bonsai=a.bonsai)
    except v41_fit.FitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    report = match_to_sprime(samples)
    report["skipped"] = skipped
    report["run_dirs"] = len(runs)
    print(format_match(report))
    print(f"run dirs {len(runs)}, skipped {skipped}")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
