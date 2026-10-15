#!/usr/bin/env python
"""Gate A driver (v4.3 Step A): arms S and S-solo on a development split with the fixed ant. No model, no network, no key.

    python scripts/v43_gate_a.py --split dev_a --seed 1 --out DIR [--limit N] [--coefficients PATH]

* Config: ``config/v4_3/world_gateA.yaml`` (F1 SimLook on, F2 and F3 off, the frozen v4.2 drug budget).
* Development cases only (``dev_a`` / ``dev_b`` of the committed split; an evaluation, reserve or unknown id cannot get in).
* Resumable: an episode whose ``record.json`` exists is skipped (same layout as ``v43_run.py``:
  ``DIR/<arm>/<case>/seed<N>/{bundle.json, record.json, cards/}``).
* Writes ``DIR/gate_a_summary.json``: per-case coverage, the mean per arm, ``S-solo - S`` and
  ``pass = mean(S-solo) >= mean(S) - 0.05``, plus the rules dict and sha, the config sha256, the code sha and the coefficient
  file sha. It prints a short table. It decides nothing beyond that flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from antelligence.experiments.v41_gate import load_checked_split  # noqa: E402
from antelligence.experiments.v42_harness import EvalRefused, load_dev_case  # noqa: E402
from antelligence.experiments.v43_harness import V43Harness, code_sha_v43  # noqa: E402
from antelligence.worlds.tumor_mri.combined import DEFAULT_COEFFICIENTS_PATH  # noqa: E402
from v43_run import DEFAULT_CASES_ROOT, atomic_write, episode_dir, load_configs, load_estimate, run_one  # noqa: E402

DEFAULT_CONFIG = REPO / "config" / "v4_3" / "world_gateA.yaml"
ARMS = ("S", "S-solo")
TOLERANCE = 0.05


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage_of(d: Path) -> float:
    return float(json.loads((d / "record.json").read_text())["metrics"]["coverage"])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("dev_a", "dev_b", "both"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="first N cases of each chosen split (smoke runs)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS_PATH)
    ap.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT if DEFAULT_CASES_ROOT.is_dir() else None)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    world_cfg, colony_cfg, rules = load_configs(args.config)
    try:
        estimate = load_estimate(rules, args.coefficients)
    except FileNotFoundError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    doc = load_checked_split(REPO)
    wanted = ("dev_a", "dev_b") if args.split == "both" else (args.split,)
    cases: List[str] = []
    for part in wanted:
        ids = list(doc[part])
        cases += ids[:args.limit] if args.limit else ids
    base = args.cases_root if args.cases_root is not None else Path("data/v4_1/cases")
    missing = [c for c in cases if not (base / c / "case.npz").is_file()]
    cases = [c for c in cases if c not in missing]
    if missing:
        print(f"note: {len(missing)} case(s) not on disk under {base}: {', '.join(missing)}", file=sys.stderr)
    h = V43Harness(world_cfg, colony_cfg, rules, estimate, with_board=True)
    per_case: Dict[str, Dict[str, Optional[float]]] = {}
    try:
        for case_id in cases:
            case = load_dev_case(case_id, root=args.cases_root)
            row: Dict[str, Optional[float]] = {}
            for arm in ARMS:
                rec = run_one(h, arm, case, args.seed, args.out)
                row[arm] = (coverage_of(episode_dir(args.out, arm, case_id, args.seed)) if rec.get("skipped")
                            else float(rec["metrics"]["coverage"]))
            per_case[case_id] = row
            print(f"{case_id:12s} S {row['S']:.3f}  S-solo {row['S-solo']:.3f}  diff {row['S-solo'] - row['S']:+.3f}")
    except EvalRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    n = len(per_case)
    mean = {a: (sum(r[a] for r in per_case.values()) / n if n else None) for a in ARMS}
    diff = None if not n else mean["S-solo"] - mean["S"]
    summary: Dict[str, Any] = {
        "study": "v4.3", "step": "A", "gate": "A", "split": args.split, "seed": args.seed, "limit": args.limit,
        "arms": list(ARMS), "n_cases": n, "cases": list(per_case), "missing_cases": missing,
        "coverage": per_case, "mean": mean, "S_solo_minus_S": diff, "tolerance": TOLERANCE,
        "pass": bool(n and mean["S-solo"] >= mean["S"] - TOLERANCE),
        "ant_rules": rules.to_dict(), "ant_rules_sha": rules.sha(),
        "config": str(args.config), "config_sha256": sha256_file(args.config),
        "code_sha": code_sha_v43(),
        "coefficients": str(args.coefficients), "coefficients_sha256": sha256_file(args.coefficients),
    }
    atomic_write(args.out / "gate_a_summary.json", json.dumps(summary, indent=1, sort_keys=True))
    if n:
        print(f"\n{'':12s} {'S':>8s} {'S-solo':>8s} {'diff':>8s}\n{'mean':12s} {mean['S']:8.3f} {mean['S-solo']:8.3f} {diff:+8.3f}"
              f"\npass (mean S-solo >= mean S - {TOLERANCE}): {summary['pass']}")
    else:
        print("no cases run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
