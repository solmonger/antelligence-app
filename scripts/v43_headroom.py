#!/usr/bin/env python
"""Headroom report (goal section 4): from a results.json with S-solo and S' on the same cases, the share where S' - S-solo >= 0.10,
with the mean and median gap. Report only; there is no pass flag and nothing is tuned from it.

    python scripts/v43_headroom.py --results runs/headroom-mid [--split dev-A] [--seeds 1] [--out headroom.json]

Reads ``results.json`` (or a directory holding it). Writes ``headroom.json`` next to it unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments.v43_drive import OutRefused, assert_out_allowed  # noqa: E402
from antelligence.experiments.v43_stats import headroom_report  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--split", choices=("dev-A", "dev-B"), default=None)
    ap.add_argument("--seeds", default="", help="comma list; default all seeds in the file")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res_path = a.results / "results.json" if a.results.is_dir() else a.results
    out = a.out or (res_path.parent / "headroom.json")
    try:
        assert_out_allowed(out)
    except OutRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] or None
    try:
        doc = headroom_report(json.loads(res_path.read_text()), seeds=seeds, split=a.split)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    doc["source"] = str(res_path)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"pairs {doc['n_pairs']} cases {doc['n_cases']}: S' - S-solo >= {doc['threshold']:.2f} on {doc['share_pairs_ge_threshold']:.3f} of pairs "
          f"({doc['share_cases_ge_threshold']:.3f} of cases); mean gap {doc['mean_gap']:.4f}, median gap {doc['median_gap']:.4f}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
