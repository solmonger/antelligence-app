"""Free check: a smarter stop rule for the finish world ("look around before you rest"). Scripted, $0, no model calls.

Background (``EARNED_TRUST.md``): most of the work a swarm leaves when it says "done" is left by its stop rule, not by
cheats. Today an ant counts itself idle after ``patience`` ticks with nothing to do, wherever it spent them, often
going back and forth over the same empty sites while work waits elsewhere. Rules (``FinishConfig.idle_rule``):

* ``ticks``  (now): idle after ``patience`` ticks with nothing to do.
* ``walk``   the ablation: same count, but while it has nothing to do an ant walks to next-door sites it has not yet
             found empty.
* ``places`` ("look around before you rest"): the same walk, and idle only after finding nothing to do at
             ``patience`` *different* sites since it last worked.

Waiting longer always leaves less work, so rules are compared **at equal time**:

1. **Match** on development seeds 0 to 9 (256 ants, 20% shortcut-takers): for each ``ticks`` patience (10 and 20),
   give ``walk`` and ``places`` the largest patience whose mean time to "done" is no longer.
2. **Test** every matched rule on fresh seeds 100 to 129, at 256 and 1,024 ants, with no shortcut-takers and with
   20%, paired by seed with the ``ticks`` rule it was matched to. Lower work left at no more time is a win.
3. **Curve**: every patience of every rule on the test seeds at 256 ants, so the whole trade-off is visible.

Two looking regimes: ``slow`` (re-checking a claim takes 3 ticks, spot looking one claim in four, the rule from section
0.2) and ``cheap`` (1 tick, look at every claim, the world's default).

Usage::

    uv run python scripts/stop_rule_check.py --out docs/research/swarm-scale-20261029/stop_rule.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from antelligence.worlds.finish import FinishConfig, FinishWorld

REGIMES = {"slow": {"look_cost": 3, "look_p": 0.25}, "cheap": {"look_cost": 1, "look_p": 1.0}}
GRID = {"ticks": (10, 15, 20, 30), "walk": (5, 8, 10, 12, 15, 20), "places": (5, 8, 10, 12, 15, 20)}
BASELINES = (10, 20)


def run_one(job: Tuple[str, int, int, float, str, int]) -> Dict[str, Any]:
    regime, n, seed, share, rule, patience = job
    cfg = FinishConfig(n_ants=n, patience=patience, idle_rule=rule, **REGIMES[regime])
    r = FinishWorld(cfg, seed).run("look", share)
    return {"regime": regime, "n": n, "seed": seed, "shortcut_share": share, "rule": rule, "patience": patience,
            "skipped_pct": round(100 * r.skipped_share, 5), "stop_tick": r.stop_tick or cfg.max_ticks,
            "open_pieces": r.open_pieces_at_stop}


def _map(jobs: List[Tuple[Any, ...]], workers: int) -> List[Dict[str, Any]]:
    jobs = sorted(set(jobs), key=lambda j: (-j[1], j))  # big swarms first, so the pool ends evenly
    if workers <= 1:
        return [run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, jobs, chunksize=2))


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def mean_of(rows: List[Dict[str, Any]], metric: str, **kw: Any) -> float:
    sel = [r[metric] for r in rows if all(r[k] == v for k, v in kw.items())]
    return round(st.mean(sel), 4)


def match(dev: List[Dict[str, Any]], regime: str) -> Dict[str, Dict[int, int]]:
    """For each baseline patience, the largest patience of each new rule that is no slower on the dev seeds."""
    out: Dict[str, Dict[int, int]] = {}
    for rule in ("walk", "places"):
        out[rule] = {}
        for base in BASELINES:
            budget = mean_of(dev, "stop_tick", regime=regime, rule="ticks", patience=base)
            ok = [p for p in GRID[rule] if mean_of(dev, "stop_tick", regime=regime, rule=rule, patience=p) <= budget]
            out[rule][base] = max(ok) if ok else min(GRID[rule])
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dev-seeds", default="0-9")
    ap.add_argument("--seeds", default="100-129")
    ap.add_argument("--ns", default="256,1024")
    ap.add_argument("--shares", default="0.0,0.2")
    ap.add_argument("--regimes", default="slow,cheap")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    def seeds(spec: str) -> List[int]:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))

    dev_seeds, test_seeds = seeds(a.dev_seeds), seeds(a.seeds)
    if set(dev_seeds) & set(test_seeds):
        raise SystemExit("test seeds must be fresh: they overlap the development seeds")
    regimes = a.regimes.split(",")
    ns = [int(x) for x in a.ns.split(",")]
    shares = [float(x) for x in a.shares.split(",")]
    small = min(ns)

    dev = _map([(g, small, s, 0.2, rule, p) for g in regimes for rule, ps in GRID.items() for p in ps
                for s in dev_seeds], a.workers)
    matched = {g: match(dev, g) for g in regimes}
    jobs = [(g, small, s, sh, rule, p) for g in regimes for rule, ps in GRID.items() for p in ps
            for s in test_seeds for sh in shares]
    for g in regimes:
        for n in ns:
            for sh in shares:
                for s in test_seeds:
                    for base in BASELINES:
                        jobs.append((g, n, s, sh, "ticks", base))
                        for rule in ("walk", "places"):
                            jobs.append((g, n, s, sh, rule, matched[g][rule][base]))
    rows = _map(jobs, a.workers)

    comps = []
    for g in regimes:
        for n in ns:
            for sh in shares:
                for base in BASELINES:
                    for rule in ("walk", "places"):
                        p = matched[g][rule][base]
                        ra = {r["seed"]: r for r in rows if (r["regime"], r["n"], r["shortcut_share"], r["rule"],
                                                             r["patience"]) == (g, n, sh, rule, p)}
                        rb = {r["seed"]: r for r in rows if (r["regime"], r["n"], r["shortcut_share"], r["rule"],
                                                             r["patience"]) == (g, n, sh, "ticks", base)}
                        c = {"regime": g, "n": n, "shortcut_share": sh, "baseline_patience": base, "rule": rule,
                             "patience": p}
                        for metric in ("skipped_pct", "stop_tick"):
                            diffs = [ra[s][metric] - rb[s][metric] for s in sorted(ra) if s in rb]
                            mean, lo, hi = boot_ci(diffs)
                            c[metric] = {"a": mean_of(list(ra.values()), metric), "b": mean_of(list(rb.values()), metric),
                                         "diff": mean, "ci95": [lo, hi], "a_lower": sum(d < 0 for d in diffs),
                                         "a_higher": sum(d > 0 for d in diffs), "seeds": len(diffs)}
                        comps.append(c)
    curve = [{"regime": g, "shortcut_share": sh, "rule": rule, "patience": p,
              "skipped_pct": mean_of(rows, "skipped_pct", regime=g, n=small, shortcut_share=sh, rule=rule, patience=p),
              "stop_tick": mean_of(rows, "stop_tick", regime=g, n=small, shortcut_share=sh, rule=rule, patience=p)}
             for g in regimes for sh in shares for rule, ps in GRID.items() for p in ps]

    print("Matched on dev seeds (rule: patience for ticks@10, ticks@20):",
          {g: {r: [m[r][b] for b in BASELINES] for r in m} for g, m in matched.items()})
    print("\nTest seeds, matched rule minus ticks rule: work left (points), ticks")
    for c in comps:
        w, t = c["skipped_pct"], c["stop_tick"]
        print(f"  {c['regime']:>5} n={c['n']:>5} cheats {c['shortcut_share']:.0%} ticks@{c['baseline_patience']:<2} vs "
              f"{c['rule']}@{c['patience']:<2}: left {w['b']:.3f}% -> {w['a']:.3f}% ({w['diff']:+.3f} "
              f"[{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}], lower on {w['a_lower']}/{w['seeds']})  "
              f"ticks {t['b']:.1f} -> {t['a']:.1f} ({t['diff']:+.1f} [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}])")
    print(f"\nCurve at {small} ants, test seeds: work left % @ ticks")
    for g in regimes:
        for sh in shares:
            for rule in GRID:
                pts = [c for c in curve if (c["regime"], c["shortcut_share"], c["rule"]) == (g, sh, rule)]
                print(f"  {g:>5} cheats {sh:.0%} {rule:>6}: " +
                      "  ".join(f"p{c['patience']}: {c['skipped_pct']:.3f}@{c['stop_tick']:.0f}" for c in pts))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "stop_rule_check", "regimes": REGIMES, "grid": GRID,
                                     "dev_seeds": dev_seeds, "test_seeds": test_seeds, "matched": matched,
                                     "comparisons": comps, "curve": curve, "dev_rows": dev, "rows": rows},
                                    indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
