"""Free check: earned trust in the finish world. Do ants that remember who they caught skipping work get cheaper,
better looking? Scripted, $0, no model calls.

Background (plan section 0.2): when re-checking a "done" claim takes 3 ticks, looking at every claim doubles the time
and still leaves work behind; spot looking (one claim in four) is faster and leaves less. The plan's next step was to
look more often at claims from ants with a poor record (earned trust, idea 1). Arms, on the same worlds:

* ``look@p``    spot looking at ``look_p`` (the current rule).
* ``record@p``  spot looking, but every claim from an ant this ant caught before is re-checked (private memory).
* ``gossip@p``  ``record``, plus ants on the same site pool the names they caught.
* ``look@1``    look at every claim (reference).

Grid: 256 and 1,024 ants, 10% and 20% shortcut-takers, patience 10 and 20 idle ticks, re-check cost 3. Fresh seeds
100 to 129 (``finish_check`` used 0 to 9). Score: share of the job's work still left when the swarm said it was done,
and ticks to "done". Paired by seed, percentile bootstrap intervals.

Usage::

    uv run python scripts/earned_trust_check.py --out docs/research/swarm-scale-20261029/earned_trust.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from antelligence.worlds.finish import FinishConfig, FinishWorld

ARMS = ("look@p", "record@p", "gossip@p", "look@1")
PAIRS = (("record@p", "look@p"), ("gossip@p", "look@p"), ("record@p", "look@1"))


def run_one(job: Tuple[int, int, float, int, float, int, str]) -> Dict[str, Any]:
    n, seed, share, patience, look_p, cost, arm = job
    name, p = arm.split("@")
    cfg = FinishConfig(n_ants=n, look_cost=cost, patience=patience, look_p=look_p if p == "p" else 1.0,
                       idle_rule="ticks")  # the stop rule this check was run with (before places became the default)
    r = FinishWorld(cfg, seed).run(name, share)
    return {"n": n, "seed": seed, "shortcut_share": share, "patience": patience, "arm": arm,
            "skipped_pct": round(100 * r.skipped_share, 5), "stop_tick": r.stop_tick or cfg.max_ticks,
            "open_pieces": r.open_pieces_at_stop, "look_ticks": r.look_ticks, "lies": r.lies,
            "caught": r.caught, "caught_wrongly": r.caught_wrongly}


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cells, comps = [], []
    keys = sorted({(r["n"], r["shortcut_share"], r["patience"]) for r in rows})
    for n, share, patience in keys:
        sel = [r for r in rows if (r["n"], r["shortcut_share"], r["patience"]) == (n, share, patience)]
        for arm in ARMS:
            p = [r for r in sel if r["arm"] == arm]
            cells.append({"n": n, "shortcut_share": share, "patience": patience, "arm": arm, "runs": len(p),
                          **{m: round(st.mean(r[m] for r in p), 3)
                             for m in ("skipped_pct", "stop_tick", "open_pieces", "look_ticks", "caught")},
                          "caught_wrongly": sum(r["caught_wrongly"] for r in p)})
        for a, b in PAIRS:
            for metric in ("skipped_pct", "stop_tick"):
                ra = {r["seed"]: r[metric] for r in sel if r["arm"] == a}
                rb = {r["seed"]: r[metric] for r in sel if r["arm"] == b}
                diffs = [ra[s] - rb[s] for s in sorted(ra) if s in rb]
                mean, lo, hi = boot_ci(diffs)
                comps.append({"n": n, "shortcut_share": share, "patience": patience, "a": a, "b": b, "metric": metric,
                              "mean_diff": mean, "ci95": [lo, hi], "seeds": len(diffs),
                              "a_higher": sum(d > 0 for d in diffs), "b_higher": sum(d < 0 for d in diffs)})
    return {"cells": cells, "comparisons": comps}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", default="256,1024")
    ap.add_argument("--shares", default="0.1,0.2")
    ap.add_argument("--patience", default="10,20")
    ap.add_argument("--look-p", type=float, default=0.25)
    ap.add_argument("--look-cost", type=int, default=3)
    ap.add_argument("--seeds", default="100-129")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    jobs = [(n, seed, share, pat, a.look_p, a.look_cost, arm)
            for n in (int(x) for x in a.ns.split(",")) for share in (float(x) for x in a.shares.split(","))
            for pat in (int(x) for x in a.patience.split(",")) for seed in range(lo, hi + 1) for arm in ARMS]
    jobs.sort(key=lambda j: -j[0])  # big swarms first, so the pool ends evenly
    if a.workers <= 1:
        rows = [run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            rows = list(pool.map(run_one, jobs, chunksize=2))
    summary = summarise(rows)
    print(f"look_p {a.look_p}, re-check cost {a.look_cost}: work left at 'done' (%), ticks, open pieces, names caught")
    for c in summary["cells"]:
        print(f"  n={c['n']:>5} cheats {c['shortcut_share']:.0%} patience {c['patience']:>2} {c['arm']:>9}: "
              f"{c['skipped_pct']:.3f}%  {c['stop_tick']:.1f} ticks  {c['open_pieces']:.1f} open  "
              f"caught {c['caught']:.0f} (wrong {c['caught_wrongly']})")
    print()
    for c in summary["comparisons"]:
        print(f"  n={c['n']:>5} cheats {c['shortcut_share']:.0%} patience {c['patience']:>2} {c['metric']:>11}: "
              f"{c['a']} - {c['b']:<8} = {c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
              f"higher {c['a_higher']}/{c['b_higher']} of {c['seeds']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "earned_trust_check", "look_p": a.look_p, "look_cost": a.look_cost,
                                     "seeds": [lo, hi], "summary": summary, "rows": rows},
                                    indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
