#!/usr/bin/env python3
"""results.json -> REPORT.md (tables only, every table with a Source line).

    python scripts/v43_report.py RESULTS [--out PATH] [--diagnostics diagnostics_numbers.json] [--effect effect.json] [--cite PATH]

Default output: ``docs/research/v4.3-20261015/REPORT.md``. ``--diagnostics`` (from ``scripts/v43_diagnose.py``) adds the
poor-tissue release and gap-by-cause tables; ``--effect`` (from ``scripts/v43_effect.py``) adds the effect rate by kind
(uptake marks apart); the realised-u table is read from ``docs/research/v4.3-20261015/uptake/realised_u.json``. ``--cite*`` name
the committed paths the numbers will live at (default: the file given).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antelligence.experiments.v43_drive import OutRefused, assert_out_allowed  # noqa: E402
from antelligence.experiments.v43_report import write_report  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "research" / "v4.3-20261015" / "REPORT.md"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--diagnostics", type=Path, default=None)
    ap.add_argument("--effect", type=Path, default=None)
    ap.add_argument("--cite", default=None, help="committed path of results.json (default: the path given)")
    ap.add_argument("--cite-diagnostics", default=None)
    ap.add_argument("--cite-effect", default=None)
    a = ap.parse_args(argv)
    try:
        assert_out_allowed(a.out)
    except OutRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    kw = {"cite": a.cite or str(a.results)}
    if a.diagnostics:
        kw.update(diagnostics=json.loads(a.diagnostics.read_text()), cite_diagnostics=a.cite_diagnostics or str(a.diagnostics))
    if a.effect:
        kw.update(effect=json.loads(a.effect.read_text()), cite_effect=a.cite_effect or str(a.effect))
    write_report(a.results, a.out, **kw)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
