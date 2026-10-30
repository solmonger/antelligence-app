"""Free check: steady eyes, the think-alone forager that checks its own eyes, in the SwarmBench foraging world. $0.

Think alone could not handle one case: an ant whose own eyes go wrong (``v2``, the view mirrored west-east from some
tick on). ``steady`` (:class:`SteadyEyesForager`) is think alone (``look``) plus a check of its eyes against the walls
it already knows; it is never told whether or when the flip comes. Arms, all on the same layouts and seeds:

* ``S_v1``     the pilot's S' for v1 (trusts its eyes, ignores talk).
* ``look``     think alone, then compare (``LookForager``), the current default.
* ``steady``   ``look`` plus the eye check.
* ``unmirror`` the pilot's S' for v2, **told** the flip tick (a reference that knows more than any ant should).

Conditions: ``base``, ``v1`` (food runs out), ``v2`` with the flip at tick 30 (the pilot's), and ``v2`` flipped at
ticks 10 and 60 (``v2@10``, ``v2@60``), so a rule tuned to one tick would show. Talk: first-hand only (``seen``) or
relayed memories too (``relay``). Fresh seeds 200 to 279 (earlier checks used 100 to 139). Score: deliveries. Also:
for ``steady``, how often an ant switched how it reads its eyes, and how many ticks after the flip it did.

``--grid`` instead runs ``steady`` alone over a grid of its two settings (how much evidence it needs before switching,
and how fast old evidence fades), to show the result does not hinge on the settings picked. Run it on the older seeds
100 to 139 (``--seeds 100-139``), which the settings were not picked on either.

Usage::

    uv run python scripts/steady_eyes_check.py --out docs/research/swarm-scale-20261029/steady_eyes.json
    uv run python scripts/steady_eyes_check.py --grid --seeds 100-139 --out docs/research/swarm-scale-20261029/steady_eyes_grid.json
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

from antelligence.kernel.field import GridField
from antelligence.kernel.scheduler import RunConfig, Scheduler
from antelligence.worlds.foraging_sb.scripts import ExploreWhenEmpty, Unmirror
from antelligence.worlds.foraging_sb.think_alone import LookForager, RelayForagingWorld, SteadyEyesForager
from antelligence.worlds.foraging_sb.world import STEPS, V1_PICKUPS, V2_FLIP_TICK

CONDITIONS = {"base": ("base", V2_FLIP_TICK), "v1": ("v1", V2_FLIP_TICK), "v2": ("v2", V2_FLIP_TICK),
              "v2@10": ("v2", 10), "v2@60": ("v2", 60)}
TALK = ("seen", "relay")
ARMS = ("S_v1", "look", "steady", "unmirror")
GRID = {"switch": (1.0, 2.0, 3.0, 4.0), "leak": (0.6, 0.8, 0.9)}


def run_one(job: Tuple[Any, ...]) -> Dict[str, Any]:
    seed, cond, talk, arm, *settings = job
    condition, flip = CONDITIONS[cond]
    world = RelayForagingWorld(seed, condition, relay=(talk == "relay"), n_pickups=V1_PICKUPS, flip_tick=flip,
                               max_steps=STEPS)
    kw = dict(settings[0]) if settings else {}
    pol = {"S_v1": ExploreWhenEmpty, "look": LookForager,
           "steady": lambda: SteadyEyesForager(**kw)}.get(arm, lambda: Unmirror(flip))()
    Scheduler(world, {}, GridField(), RunConfig(f"forage-{cond}-{seed}", seed=seed, max_ticks=STEPS),
              default_policy=pol, record_frames=False).run()
    row = {"seed": seed, "condition": cond, "talk": talk, "arm": arm, "deliveries": world.deliveries, **kw}
    if arm == "steady":
        s = pol.sensor.stats()
        row.update(switches=s["switches"], mirrored_ticks=s["mirrored_ticks"],
                   switch_delays=[t - flip for t in s["switch_ticks"]] if condition == "v2" else [],
                   false_switches=sum(t < flip for t in s["switch_ticks"]) if condition == "v2" else s["switches"])
    return row


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 3), round(means[int(0.025 * reps)], 3), round(means[int(0.975 * reps) - 1], 3)


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cells, comps = [], []
    for cond in CONDITIONS:
        for talk in TALK:
            sel = [r for r in rows if r["condition"] == cond and r["talk"] == talk]
            if not sel:
                continue
            for arm in ARMS:
                p = [r for r in sel if r["arm"] == arm]
                cell = {"condition": cond, "talk": talk, "arm": arm, "runs": len(p),
                        "deliveries": round(st.mean(r["deliveries"] for r in p), 2)}
                if arm == "steady":
                    delays = [d for r in p for d in r["switch_delays"]]
                    cell.update(switches_per_run=round(st.mean(r["switches"] for r in p), 2),
                                false_switches=sum(r["false_switches"] for r in p),
                                median_switch_delay=st.median(delays) if delays else None,
                                max_switch_delay=max(delays) if delays else None)
                cells.append(cell)
            for a, b in (("steady", "look"), ("steady", "S_v1"), ("steady", "unmirror")):
                ra = {r["seed"]: r["deliveries"] for r in sel if r["arm"] == a}
                rb = {r["seed"]: r["deliveries"] for r in sel if r["arm"] == b}
                diffs = [ra[s] - rb[s] for s in sorted(ra) if s in rb]
                mean, lo, hi = boot_ci(diffs)
                comps.append({"condition": cond, "talk": talk, "a": a, "b": b, "mean_diff": mean, "ci95": [lo, hi],
                              "seeds": len(diffs), "a_higher": sum(d > 0 for d in diffs),
                              "b_higher": sum(d < 0 for d in diffs)})
    return {"cells": cells, "comparisons": comps}


def grid_summary(rows: List[Dict[str, Any]], conditions: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for switch in GRID["switch"]:
        for leak in GRID["leak"]:
            cell: Dict[str, Any] = {"switch": switch, "leak": leak}
            for cond in conditions:
                p = [r for r in rows if r["condition"] == cond and r["switch"] == switch and r["leak"] == leak]
                delays = [d for r in p for d in r["switch_delays"]]
                cell[cond] = {"deliveries": round(st.mean(r["deliveries"] for r in p), 2),
                              "false_switches": sum(r["false_switches"] for r in p),
                              "median_switch_delay": st.median(delays) if delays else None}
            out.append(cell)
    return out


def _map(jobs: List[Tuple[Any, ...]], workers: int) -> List[Dict[str, Any]]:
    if workers <= 1:
        return [run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, jobs, chunksize=4))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="200-279")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grid", action="store_true", help="steady alone over a grid of its settings, first-hand talk")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    conditions = a.conditions.split(",")
    if a.grid:
        settings = [(("switch", sw), ("leak", lk)) for sw in GRID["switch"] for lk in GRID["leak"]]
        rows = _map([(seed, cond, "seen", "steady", kw) for cond in conditions for seed in range(lo, hi + 1)
                     for kw in settings], a.workers)
        table = grid_summary(rows, conditions)
        print("steady over its settings: deliveries (false switches, median delay) per condition")
        for c in table:
            print(f"  switch {c['switch']:.0f} leak {c['leak']:.1f}: " + "  ".join(
                f"{cond} {c[cond]['deliveries']:.2f} ({c[cond]['false_switches']}, {c[cond]['median_switch_delay']})"
                for cond in conditions))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({"check": "steady_eyes_check --grid", "conditions": CONDITIONS, "grid": GRID,
                                         "seeds": [lo, hi], "table": table, "rows": rows},
                                        indent=1, sort_keys=True) + "\n")
        return 0
    jobs = [(seed, cond, talk, arm) for cond in conditions for talk in TALK
            for seed in range(lo, hi + 1) for arm in ARMS]
    rows = _map(jobs, a.workers)
    summary = summarise(rows)
    print(f"{'cond':>6} {'talk':>5} {'arm':>9} | deliveries | eye switches per run, false, median/max delay")
    for c in summary["cells"]:
        extra = (f" | {c['switches_per_run']:.2f}, {c['false_switches']}, {c['median_switch_delay']}/{c['max_switch_delay']}"
                 if c["arm"] == "steady" else "")
        print(f"{c['condition']:>6} {c['talk']:>5} {c['arm']:>9} | {c['deliveries']:>6.2f}{extra}")
    print()
    for c in summary["comparisons"]:
        print(f"{c['condition']:>6} {c['talk']:>5}: {c['a']} - {c['b']:<8} = {c['mean_diff']:+.3f} "
              f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  higher {c['a_higher']}/{c['b_higher']} of {c['seeds']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "steady_eyes_check", "conditions": CONDITIONS, "summary": summary,
                                     "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
