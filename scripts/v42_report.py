#!/usr/bin/env python3
"""results.json -> docs/research/v4.2-20261014/REPORT.md (tables only). Usage: v42_report.py RESULTS [--out PATH]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antelligence.experiments.v42_report import write_report  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "research" / "v4.2-20261014" / "REPORT.md"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    write_report(a.results, a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
