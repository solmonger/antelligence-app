#!/usr/bin/env python
"""Replication verdict (goal section 4): the pooled 80 pairs (40 cases x seeds 7 and 8) must meet the FULL v4.2 bar for the one
comparison that passed the primary verdict.

    python scripts/v43_replicate.py --results runs/replication-S-marks --arm S-marks [--seeds 7,8] [--resamples 10000]

``--results`` is a directory holding ``results.json`` (written by ``scripts/v43_verdict.py`` with ``--arms S-solo,<arm>,<control>``
on seeds 7 and 8). The bar: median paired difference vs S-solo >= 0.03; Holm-adjusted p < 0.05 with the other primary at p = 1
(adjusted p = min(1, 2p)); the same sign on dev-A and dev-B; and the arm beats its own control (median > 0, interval's lower end
> 0). A replication with other than 80 pairs fails. Writes ``replication.json`` into ``--results`` (or ``--out``). Reads only.
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
from antelligence.experiments.v43_stats import CONTROL_OF, REPLICATION_PAIRS, replication_verdict  # noqa: E402
from antelligence.experiments.v42_stats import N_RESAMPLES  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path, help="directory with results.json (or the results.json file)")
    ap.add_argument("--arm", required=True, choices=sorted(CONTROL_OF))
    ap.add_argument("--seeds", default="7,8")
    ap.add_argument("--expected-pairs", type=int, default=REPLICATION_PAIRS)
    ap.add_argument("--resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--out", type=Path, default=None, help="default: <results dir>/replication.json")
    a = ap.parse_args(argv)
    res_path = a.results / "results.json" if a.results.is_dir() else a.results
    out = a.out or (res_path.parent / "replication.json")
    try:
        assert_out_allowed(out)
    except OutRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    results = json.loads(res_path.read_text())
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    try:
        doc = replication_verdict(results, a.arm, CONTROL_OF[a.arm], n=a.resamples, expected_pairs=a.expected_pairs, seeds=seeds)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    doc["source"] = str(res_path)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    v, b = doc["verdict"], doc["boot"]
    print(f"{doc['comparison']}: pairs {doc['n_pairs']} (expected {a.expected_pairs}), median {b['median']:.4f} "
          f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}], holm p {doc['holm_p']:.4f}, dev-A {doc['by_split']['dev-A']} dev-B {doc['by_split']['dev-B']}")
    print(f"beats {doc['control']}: median {doc['control_boot']['median']:.4f}, CI low {doc['control_boot']['ci_low']:.4f}")
    print("conditions: " + ", ".join(f"{k}={v[k]}" for k in ("median_ge_0.03", "holm_p_lt_0.05", "same_sign_dev_a_b", "beats_control", "pairs_complete")))
    print(f"REPLICATION {'PASS' if doc['pass'] else 'FAIL'}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
