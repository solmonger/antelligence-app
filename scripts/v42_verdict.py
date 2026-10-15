#!/usr/bin/env python
"""Phase 0 verdict driver (goal 9.2-9.3). All arms, the frozen config, development cases only.

    python scripts/v42_verdict.py --out runs/verdict --cases BRATS_016,BRATS_406 --seeds 1 [--workers 4]
    python scripts/v42_verdict.py --out runs/verdict --dev-a 30 --dev-b 10 --seeds 1,2,3

Arms (default): S, S-solo, S-marks, S-talk, S-marks-displaced, S-talk-shuffled, S', S-marks-perfect (diagnostic),
S-marks-quiet (quiet pair). Frozen config: ``config/v4_2/world.yaml``; its fact flags must be on (the lead freezes them after
tuning). ``--f2 U_CORE`` / ``--f3 DECAY,SHARE`` override them for dry runs only and are recorded in ``results.json`` meta.

Resumable (an episode whose ``record.json`` is complete is skipped), process pool of at most 6 workers, then: assemble
``results.json`` -> M1 analyse -> ``REPORT.md`` -> S' ceiling check (``verdict_checks.json``). Nothing is written under
``docs/`` or ``config/``; ``--report`` can point the report elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from antelligence.experiments import v42_drive as drive  # noqa: E402
from antelligence.experiments.v41_gate import load_checked_split  # noqa: E402
from antelligence.experiments.v42_harness import EvalRefused, load_dev_case, resolve_arm  # noqa: E402
from antelligence.experiments.v42_report import render  # noqa: E402
from antelligence.experiments.v42_results import write_results  # noqa: E402

VERDICT_ARMS = ("S", "S-solo", "S-marks", "S-talk", "S-marks-displaced", "S-talk-shuffled", "S'")
EXTRA_ARMS = ("S-marks-perfect", "S-marks-quiet")
DEFAULT_ARMS = VERDICT_ARMS + EXTRA_ARMS
CEILING_SHARE = 0.95


def ceiling_check(runs: List[Mapping[str, Any]], arms: List[str]) -> Dict[str, Any]:
    """S' >= every verdict arm on >= 95% of case-seeds (goal 9.3 sanity). Logged, never fatal."""
    by: Dict[tuple, Dict[str, float]] = {}
    for r in runs:
        by.setdefault((r["case_id"], r["seed"]), {})[r["arm"]] = float(r["coverage"])
    others = [a for a in arms if a in VERDICT_ARMS and a != "S'"]
    keys = [k for k, v in by.items() if "S'" in v]
    viol = []
    for k in sorted(keys):
        worse = [a for a in others if a in by[k] and by[k][a] > by[k]["S'"] + 1e-12]
        if worse:
            viol.append({"case_id": k[0], "seed": k[1], "s_prime": by[k]["S'"], "beaten_by": {a: by[k][a] for a in worse}})
    share = None if not keys else 1.0 - len(viol) / len(keys)
    return {"n_case_seeds": len(keys), "share_s_prime_ge_all": share, "required": CEILING_SHARE,
            "ok": share is not None and share >= CEILING_SHARE, "violations": viol}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", default="", help="comma list of development case ids")
    ap.add_argument("--dev-a", type=int, default=0, help="take the first N dev-A cases")
    ap.add_argument("--dev-b", type=int, default=0, help="take the first N dev-B cases")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--workers", type=int, default=4, help=f"at most {drive.MAX_WORKERS}")
    ap.add_argument("--config", type=Path, default=drive.DEFAULT_CONFIG)
    ap.add_argument("--cases-root", type=Path, default=drive.DEFAULT_CASES_ROOT if drive.DEFAULT_CASES_ROOT.is_dir() else None)
    ap.add_argument("--ticks", type=int, default=None, help="override episode length (dry runs only)")
    ap.add_argument("--f2", type=float, default=None, metavar="U_CORE", help="dry-run override of the frozen F2")
    ap.add_argument("--f3", type=lambda s: tuple(float(x) for x in s.split(",")), default=None, metavar="DECAY,SHARE")
    ap.add_argument("--report", type=Path, default=None, help="REPORT.md path (default: <out>/REPORT.md)")
    ap.add_argument("--resamples", type=int, default=None, help="bootstrap resamples (default: M1's)")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    split = load_checked_split(REPO)
    cases = [c.strip() for c in a.cases.split(",") if c.strip()] + split["dev_a"][:a.dev_a] + split["dev_b"][:a.dev_b]
    cases = list(dict.fromkeys(cases))
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    try:
        arms = [resolve_arm(x.strip()).name for x in a.arms.split(",") if x.strip()]
        drive.guard_dev_cases(cases)
    except (EvalRefused, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if not cases or not seeds:
        print("no cases or seeds given", file=sys.stderr)
        return 2
    if a.workers > drive.MAX_WORKERS:
        print(f"refused: at most {drive.MAX_WORKERS} workers", file=sys.stderr)
        return 2
    wc, cc = drive.load_configs(a.config)
    if not wc.facts.any_on and a.f2 is None and a.f3 is None:
        print("refused: the config has every hidden fact off; freeze it first (or pass --f2/--f3 for a dry run)", file=sys.stderr)
        return 2
    tasks = [{"kind": "episode", "arm": arm, "case_id": c, "seed": s, "config": str(a.config), "out": str(a.out),
              "cases_root": str(a.cases_root) if a.cases_root else None, "ticks": a.ticks, "board": True,
              "facts": {"f2": a.f2, "f3": a.f3}} for c in cases for s in seeds for arm in arms]
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    times: List[Dict[str, Any]] = []
    print(f"{len(tasks)} episodes ({len(arms)} arms x {len(cases)} cases x {len(seeds)} seeds), {a.workers} workers", flush=True)

    def on_result(task: Mapping[str, Any], r: Dict[str, Any]) -> None:
        if r.get("skipped"):
            print(f"  skip {task['arm']:18s} {task['case_id']:10s} seed {task['seed']} (done)", flush=True)
            return
        times.append({"arm": r["arm"], "case_id": r["case_id"], "seed": r["seed"], **r["timing_s"]})
        print(f"  ran  {r['arm']:18s} {r['case_id']:10s} seed {r['seed']}  coverage {r['coverage']:.3f}  {r['timing_s']['wall']:.2f}s", flush=True)

    drive.run_pool(tasks, a.workers, on_result)

    def truth_for(case_id: str):
        return load_dev_case(case_id, root=a.cases_root).crop.truth

    meta = {"config": str(a.config), "frozen_facts": wc.facts.__dict__, "dry_overrides": {"f2": a.f2, "f3": a.f3, "ticks": a.ticks},
            "workers": a.workers, "seeds": seeds, "arms": arms}
    results = write_results(a.out, truth_for=truth_for, meta=meta)
    results["meta"]["ceiling_check"] = chk = ceiling_check(results["runs"], arms)
    (a.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=str))
    (a.out / "verdict_checks.json").write_text(json.dumps({"ceiling": chk, "episode_times": times}, indent=1, sort_keys=True))
    from antelligence.experiments.v42_report import compute_stats
    stats = compute_stats(results, **({"n": a.resamples} if a.resamples else {}))
    text = render(results, stats)
    report = a.report or (a.out / "REPORT.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text)
    if stats is not None:
        (a.out / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True, default=str))
    if not chk["ok"]:
        print(f"LOG: S' >= every verdict arm on only {chk['share_s_prime_ge_all']} of {chk['n_case_seeds']} case-seeds "
              f"(need {CEILING_SHARE}); see verdict_checks.json", file=sys.stderr)
    if times:
        w = sorted(t["wall"] for t in times)
        print(f"episodes run {len(w)}; wall per episode median {w[len(w) // 2]:.2f}s max {w[-1]:.2f}s; total {time.perf_counter() - t0:.1f}s")
    print(f"results {a.out / 'results.json'}\nreport  {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
