"""Free check: do shortcut-taking ants get away with skipping the last bit of a job? Scripted, $0, no model calls.

Grid: swarm size x arm (trust the tick-box, look yourself, trust plus inspectors) x fade rule (fixed, smart) x share
of shortcut-takers (0, 5%, 10%, 20%) x development seeds. Every arm runs on the same world per seed (same work, graph,
start sites and the same shortcut-takers), so arms compare pair by pair.

Reported per cell: when the swarm called the job finished, the share of the job's work still left then (work called
done but never done), unfinished pieces, and the ant-time spent working and re-checking. Key paired comparisons get a
percentile bootstrap interval over seeds. Results feed section 0.2 of docs/plans/2026-10-29-swarm-at-scale.md.

Usage::

    uv run python scripts/finish_check.py --ns 256 --out docs/research/swarm-scale-20261029/finish_n256.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from antelligence.worlds.finish import FinishConfig, FinishWorld

ARMS = [("trust", "fixed"), ("trust", "smart"), ("look", "fixed"), ("look", "smart"), ("inspect", "fixed")]
SHARES = (0.0, 0.05, 0.1, 0.2)


def run_grid(ns: Sequence[int], seeds: Sequence[int], shares: Sequence[float], base: FinishConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in ns:
        for seed in seeds:
            for fade in ("fixed", "smart"):
                world = FinishWorld(replace(base, n_ants=n, smart=(fade == "smart")), seed)
                for arm, f in ARMS:
                    if f != fade:
                        continue
                    for share in shares:
                        r = world.run(arm, share)
                        rows.append({"n": n, "seed": seed, "arm": arm, "fade": fade, "shortcut_share": share,
                                     "stop_tick": r.stop_tick, "skipped_share": r.skipped_share,
                                     "left_at_stop": r.left_at_stop, "total_work": r.total_work,
                                     "open_pieces_at_stop": r.open_pieces_at_stop, "said_done_early": r.said_done_early,
                                     "finished_tick": r.finished_tick, "work_ticks": r.work_ticks,
                                     "look_ticks": r.look_ticks, "lies": r.lies})
    return rows


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def pick(rows, **kw) -> Dict[int, Dict[str, Any]]:
    return {r["seed"]: r for r in rows if all(r[k] == v for k, v in kw.items())}


def _stop(r: Dict[str, Any], cap: int) -> int:
    return r["stop_tick"] if r["stop_tick"] is not None else cap


def compare(rows, n: int, share: float, a: Tuple[str, str], b: Tuple[str, str], metric: str, cap: int) -> Dict[str, Any]:
    ra = pick(rows, n=n, shortcut_share=share, arm=a[0], fade=a[1])
    rb = pick(rows, n=n, shortcut_share=share, arm=b[0], fade=b[1])
    get = (lambda r: _stop(r, cap)) if metric == "stop_tick" else (lambda r: r[metric])
    diffs = [get(ra[s]) - get(rb[s]) for s in sorted(ra) if s in rb]
    mean, lo, hi = boot_ci(diffs)
    return {"n": n, "shortcut_share": share, "metric": metric, "a": "/".join(a), "b": "/".join(b), "mean_diff": mean,
            "ci95": [lo, hi], "seeds": len(diffs), "a_higher": sum(d > 0 for d in diffs), "b_higher": sum(d < 0 for d in diffs)}


def summarise(rows, ns: Sequence[int], shares: Sequence[float], cap: int) -> Dict[str, Any]:
    cells = []
    for n in ns:
        for share in shares:
            for arm, fade in ARMS:
                sel = [r for r in rows if r["n"] == n and r["shortcut_share"] == share and r["arm"] == arm and r["fade"] == fade]
                if not sel:
                    continue
                cells.append({"n": n, "shortcut_share": share, "arm": arm, "fade": fade, "runs": len(sel),
                              "skipped_share": round(st.mean(r["skipped_share"] for r in sel), 4),
                              "open_pieces_at_stop": round(st.mean(r["open_pieces_at_stop"] for r in sel), 2),
                              "said_done_early": sum(r["said_done_early"] for r in sel),
                              "never_stopped": sum(r["stop_tick"] is None for r in sel),
                              "stop_tick": round(st.mean(_stop(r, cap) for r in sel), 1),
                              "work_ticks": round(st.mean(r["work_ticks"] for r in sel), 1),
                              "look_ticks": round(st.mean(r["look_ticks"] for r in sel), 1),
                              "lies": round(st.mean(r["lies"] for r in sel), 1)})
    comps = []
    for n in ns:
        for share in shares:
            for metric in ("skipped_share", "stop_tick"):
                comps.append(compare(rows, n, share, ("trust", "fixed"), ("look", "fixed"), metric, cap))
                comps.append(compare(rows, n, share, ("trust", "fixed"), ("trust", "smart"), metric, cap))
                comps.append(compare(rows, n, share, ("trust", "fixed"), ("look", "smart"), metric, cap))
                comps.append(compare(rows, n, share, ("trust", "fixed"), ("inspect", "fixed"), metric, cap))
    return {"cells": cells, "comparisons": comps}


def print_tables(summary: Dict[str, Any]) -> None:
    print(f"{'n':>5} {'cheat':>5} {'arm':>14} | {'skipped':>8} {'open':>6} {'early':>5} | {'stop':>6} {'work':>8} {'look':>8} {'lies':>6}")
    for c in summary["cells"]:
        print(f"{c['n']:>5} {c['shortcut_share']:>5.2f} {c['arm'] + '/' + c['fade']:>14} | {c['skipped_share']:>8.4f} "
              f"{c['open_pieces_at_stop']:>6.1f} {c['said_done_early']:>2}/{c['runs']:<2} | {c['stop_tick']:>6.1f} "
              f"{c['work_ticks']:>8.1f} {c['look_ticks']:>8.1f} {c['lies']:>6.1f}")
    print()
    for c in summary["comparisons"]:
        print(f"n={c['n']:>5} cheat={c['shortcut_share']:.2f} {c['metric']:>13}: {c['a']:>13} - {c['b']:<13} = "
              f"{c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  a higher {c['a_higher']}/{c['seeds']}")


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", default="64,256")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--shares", default=",".join(str(s) for s in SHARES))
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--sites-per-ant", type=float, default=4.0)
    ap.add_argument("--tail", type=float, default=0.2)
    ap.add_argument("--look-cost", type=int, default=1)
    ap.add_argument("--look-p", type=float, default=1.0)
    ap.add_argument("--max-ticks", type=int, default=600)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    ns = [int(x) for x in a.ns.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    shares = [float(x) for x in a.shares.split(",")]
    base = FinishConfig(patience=a.patience, sites_per_ant=a.sites_per_ant, tail=a.tail, look_cost=a.look_cost,
                        look_p=a.look_p, max_ticks=a.max_ticks)
    rows = run_grid(ns, seeds, shares, base)
    summary = summarise(rows, ns, shares, base.max_ticks)
    print_tables(summary)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        cfg = {k: v for k, v in base.__dict__.items() if k not in ("n_ants", "smart")}
        a.out.write_text(json.dumps({"check": "finish_check", "config": cfg, "ns": ns, "seeds": seeds, "shares": shares,
                                     "summary": summary, "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
