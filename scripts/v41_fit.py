#!/usr/bin/env python
"""Phase 0 fit (goal 5 steps 7, 9, 10, 12): calibration logs -> frozen conformal thresholds + kNN router + power + throughput.

  .venv/bin/python scripts/v41_fit.py RUN_DIR [RUN_DIR ...] --cells cells.jsonl --tag calib1

Reads development calibration runs only (an evaluation-case or non-calibration log is refused). Freezes
config/v4_1/conformal.json and config/v4_1/router.json (each with its sha256; an existing file needs --force) and writes
docs/research/v4.1-20261013/phase0/fit_<tag>.json (aggregates only). ``--bonsai`` fits the Bonsai variant (d1 / 4B /
Bonsai 8B, labels by Bonsai's answers) into conformal_bonsai.json / router_bonsai.json. ``--no-freeze`` reports only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v41_fit  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run output dirs (or dirs containing runs/*/decisions.jsonl)")
    ap.add_argument("--cells", type=Path, help="cells.jsonl (default: each run's cell.json)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bonsai", action="store_true")
    ap.add_argument("--no-freeze", action="store_true", help="report only; write no frozen files")
    ap.add_argument("--out-conformal", type=Path)
    ap.add_argument("--out-router", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--split-file", type=Path, default=REPO / "config/v4_1/split.json")
    ap.add_argument("--concurrency", type=int, help="episode concurrency of the calibration run (effective throughput)")
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--force", action="store_true", help="overwrite existing frozen files")
    a = ap.parse_args(argv)
    suffix = "_bonsai" if a.bonsai else ""
    conf = None if a.no_freeze else (a.out_conformal or REPO / f"config/v4_1/conformal{suffix}.json")
    rout = None if a.no_freeze else (a.out_router or REPO / f"config/v4_1/router{suffix}.json")
    report = a.report or REPO / f"docs/research/v4.1-20261013/phase0/fit_{a.tag}{suffix}.json"
    try:
        out = v41_fit.run_fit(a.runs, tag=a.tag, cells_path=a.cells, bonsai=a.bonsai, out_conformal=conf, out_router=rout,
                              report_path=report, split_file=a.split_file, concurrency=a.concurrency, n_eval=a.n_eval,
                              force=a.force)
    except v41_fit.FitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    c, r, p, t = out["conformal"], out["router"], out["power"], out["throughput"]
    print(f"decisions {out['inputs']['usable_decisions']}/{out['inputs']['decisions']} from {out['inputs']['run_dirs']} runs")
    print(f"conformal: joint defer {c['joint_defer_rate']:.3f} on_target={c['joint_on_target']} sha {c['sha256'][:12]}")
    print("router:", r.get("error") or f"cv agreement {r['cross_validated'].get('agreement')} shares {r['in_sample_pick_shares']}")
    print(f"power: paired sd {p['dev_paired_sd']} -> mdd_p1 {p['mdd_p1_points']} points at n={p['eval_cases']}")
    print(f"throughput: {t['decisions_per_second_per_run']} dec/s/run, {t['seconds_per_episode_mean']} s/episode")
    print(f"report: {report}")
    return 0 if out["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
