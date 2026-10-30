"""Free check of smart fading memory and think-alone ants in the rumour world. Scripted, $0, no model calls.

Grid: swarm size x fade rule (fixed, smart) x policy (solo, herd, alone) x what is planted (none, one wrong mark,
a burst of wrong marks, liars) x development seeds. Every seeded run is paired with its own no-seed control on the same random streams.

Reported per cell: share of honest ants right and wrong at the end; for seeded runs, the extra wrong share over the
control (mean over ticks and peak), and when the planted mark fell below half a reading. Key paired comparisons get
a percentile bootstrap interval over seeds.

Usage::

    uv run python scripts/fading_alone_check.py --out docs/research/swarm-scale-20261029/fading_alone.json
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

from antelligence.worlds.rumour import RumourConfig, RumourWorld

POLICIES = ("solo", "herd", "alone")
SEED_KINDS = ("none", "planted", "burst", "liars")
ARM_ORDER = [("fixed", "herd"), ("smart", "herd"), ("fixed", "alone"), ("smart", "alone"), ("fixed", "solo")]


def run_grid(ns: Sequence[int], seeds: Sequence[int], base: RumourConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in ns:
        for seed in seeds:
            for fade in ("fixed", "smart"):
                world = RumourWorld(replace(base, n_ants=n, smart=(fade == "smart")), seed)
                for policy in POLICIES:
                    if policy == "solo" and fade == "smart":
                        continue  # solo reads no marks; the fade rule cannot matter
                    control = world.run(policy, "none")
                    for kind in SEED_KINDS:
                        r = control if kind == "none" else world.run(policy, kind)
                        extra = [w - c for w, c in zip(r.wrong_by_tick, control.wrong_by_tick)]
                        rows.append({"n": n, "seed": seed, "fade": fade, "policy": policy, "seeded": kind,
                                     "right": r.right, "wrong": r.wrong, "undecided": r.undecided,
                                     "extra_wrong_mean": round(st.mean(extra), 6), "extra_wrong_peak": round(max(extra), 6),
                                     "planted_gone_tick": r.planted_gone_tick, "mean_half_life": r.mean_half_life})
    return rows


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def pick(rows, **kw) -> Dict[int, Dict[str, Any]]:
    return {r["seed"]: r for r in rows if all(r[k] == v for k, v in kw.items())}


def compare(rows, n: int, seeded: str, a: Tuple[str, str], b: Tuple[str, str], metric: str) -> Dict[str, Any]:
    ra = pick(rows, n=n, seeded=seeded, fade=a[0], policy=a[1])
    rb = pick(rows, n=n, seeded=seeded, fade=b[0], policy=b[1])
    diffs = [ra[s][metric] - rb[s][metric] for s in sorted(ra) if s in rb]
    mean, lo, hi = boot_ci(diffs)
    return {"n": n, "seeded": seeded, "metric": metric, "a": "/".join(a), "b": "/".join(b), "mean_diff": mean,
            "ci95": [lo, hi], "seeds": len(diffs), "a_wins": sum(d > 0 for d in diffs), "b_wins": sum(d < 0 for d in diffs)}


def summarise(rows, ns: Sequence[int]) -> Dict[str, Any]:
    cells = []
    for n in ns:
        for kind in SEED_KINDS:
            for fade, policy in ARM_ORDER:
                sel = [r for r in rows if r["n"] == n and r["seeded"] == kind and r["fade"] == fade and r["policy"] == policy]
                if not sel:
                    continue
                gone = [r["planted_gone_tick"] for r in sel if r["planted_gone_tick"] is not None]
                cells.append({"n": n, "seeded": kind, "fade": fade, "policy": policy, "runs": len(sel),
                              "right": round(st.mean(r["right"] for r in sel), 4),
                              "wrong": round(st.mean(r["wrong"] for r in sel), 4),
                              "extra_wrong_mean": round(st.mean(r["extra_wrong_mean"] for r in sel), 4),
                              "extra_wrong_peak": round(st.mean(r["extra_wrong_peak"] for r in sel), 4),
                              "planted_gone_median": (st.median(gone) if gone else None),
                              "planted_never_gone": sum(r["planted_gone_tick"] is None for r in sel) if kind in ("planted", "burst") else 0})
    comps = []
    for n in ns:
        comps.append(compare(rows, n, "none", ("fixed", "alone"), ("fixed", "herd"), "right"))
        comps.append(compare(rows, n, "none", ("smart", "herd"), ("fixed", "herd"), "right"))
        comps.append(compare(rows, n, "none", ("smart", "alone"), ("fixed", "alone"), "right"))
        comps.append(compare(rows, n, "none", ("smart", "alone"), ("fixed", "herd"), "right"))
        comps.append(compare(rows, n, "none", ("fixed", "herd"), ("fixed", "solo"), "right"))
        for kind in ("planted", "burst", "liars"):
            comps.append(compare(rows, n, kind, ("fixed", "herd"), ("fixed", "alone"), "extra_wrong_mean"))
            comps.append(compare(rows, n, kind, ("fixed", "herd"), ("smart", "herd"), "extra_wrong_mean"))
            comps.append(compare(rows, n, kind, ("fixed", "herd"), ("smart", "alone"), "extra_wrong_mean"))
    return {"cells": cells, "comparisons": comps}


def print_tables(summary: Dict[str, Any]) -> None:
    print(f"{'n':>5} {'seeded':>8} {'arm':>12} | {'right':>6} {'wrong':>6} | {'extra wrong (mean/peak)':>24} | planted gone")
    for c in summary["cells"]:
        gone = "" if c["seeded"] not in ("planted", "burst") else f"t={c['planted_gone_median']} (never: {c['planted_never_gone']})"
        print(f"{c['n']:>5} {c['seeded']:>8} {c['fade'] + '/' + c['policy']:>12} | {c['right']:>6.3f} {c['wrong']:>6.3f} | "
              f"{c['extra_wrong_mean']:>11.4f} {c['extra_wrong_peak']:>11.4f} | {gone}")
    print()
    for c in summary["comparisons"]:
        print(f"n={c['n']:>5} {c['seeded']:>8} {c['metric']:>17}: {c['a']:>12} - {c['b']:<12} = {c['mean_diff']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  wins {c['a_wins']}/{c['seeds']}")


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", default="64,256,1024")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--half-life", type=float, default=10.0)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--accuracy", type=float, default=0.6)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    ns = [int(x) for x in a.ns.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    base = RumourConfig(half_life=a.half_life, q=a.q, accuracy=a.accuracy, ticks=a.ticks)
    rows = run_grid(ns, seeds, base)
    summary = summarise(rows, ns)
    print_tables(summary)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        cfg = {k: v for k, v in base.__dict__.items() if k not in ("n_ants", "smart")}
        a.out.write_text(json.dumps({"check": "fading_alone_check", "config": cfg, "ns": ns, "seeds": seeds,
                                     "summary": summary, "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
