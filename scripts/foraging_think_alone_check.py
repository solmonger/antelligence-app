"""Free check: think alone, done included, in the SwarmBench foraging world. Scripted, $0, no model calls.

The world is the pilot's port of SwarmBench's foraging task (arXiv 2505.04364), a task we did not design, at its
published size (10 agents, 10x10 map, 100 steps). Conditions: ``base``; ``v1`` (the food patch runs out after 6
pickups and a fresh one appears across the lane); ``v2`` (from tick 30 every agent's view is mirrored west-east).
Talk: ``seen`` (agents broadcast only food or a nest they see, the pilot as published) or ``relay`` (agents who see
nothing also broadcast their remembered spot, which can be stale). Policies:

* ``S``       the pilot's script: trusts its brief, ignores what it hears.
* ``S_v1``    the pilot's S' written for v1: forgets a remembered spot it sees empty; ignores what it hears.
* ``look``    think alone, then compare (``LookForager``): S_v1 by default in every condition, plus it follows a
              peer's lead only when it has nothing of its own, and drops the lead when its own eyes disagree.

Score: deliveries. Every arm runs on the same layouts and seeds; seeds 100 to 139 (the pilot's evaluation seeds,
1 to 5, stay untouched).

Usage::

    uv run python scripts/foraging_think_alone_check.py --out docs/research/swarm-scale-20261029/foraging.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from antelligence.kernel.field import GridField
from antelligence.kernel.scheduler import RunConfig, Scheduler
from antelligence.worlds.foraging_sb.scripts import ExploreWhenEmpty, ShortestPath
from antelligence.worlds.foraging_sb.think_alone import LookForager, RelayForagingWorld
from antelligence.worlds.foraging_sb.world import STEPS, V1_PICKUPS, V2_FLIP_TICK

POLICIES = {"S": ShortestPath, "S_v1": ExploreWhenEmpty, "look": LookForager}
CONDITIONS = ("base", "v1", "v2")
TALK = ("seen", "relay")


def run_one(seed: int, condition: str, talk: str, policy: str) -> Dict[str, Any]:
    world = RelayForagingWorld(seed, condition, relay=(talk == "relay"), n_pickups=V1_PICKUPS,
                               flip_tick=V2_FLIP_TICK, max_steps=STEPS)
    pol = POLICIES[policy]()
    Scheduler(world, {}, GridField(), RunConfig(f"forage-{condition}-{seed}", seed=seed, max_ticks=STEPS),
              default_policy=pol, record_frames=False).run()
    row = {"seed": seed, "condition": condition, "talk": talk, "policy": policy, "deliveries": world.deliveries,
           "relayed": world.relayed}
    if isinstance(pol, LookForager):
        row.update(pol.lead_stats())
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
            for pol in POLICIES:
                p = [r for r in sel if r["policy"] == pol]
                if p:
                    cell = {"condition": cond, "talk": talk, "policy": pol, "runs": len(p),
                            "deliveries": round(st.mean(r["deliveries"] for r in p), 2)}
                    if "leads" in p[0]:
                        cell["leads"] = round(st.mean(r["leads"] for r in p), 2)
                        cell["dead_leads"] = round(st.mean(r["dead_leads"] for r in p), 2)
                    cells.append(cell)
            for a, b in (("look", "S"), ("look", "S_v1")):
                ra = {r["seed"]: r for r in sel if r["policy"] == a}
                rb = {r["seed"]: r for r in sel if r["policy"] == b}
                diffs = [ra[s]["deliveries"] - rb[s]["deliveries"] for s in sorted(ra) if s in rb]
                if diffs:
                    mean, lo, hi = boot_ci(diffs)
                    comps.append({"condition": cond, "talk": talk, "a": a, "b": b, "mean_diff": mean, "ci95": [lo, hi],
                                  "seeds": len(diffs), "a_wins": sum(d > 0 for d in diffs),
                                  "b_wins": sum(d < 0 for d in diffs)})
        ra = {r["seed"]: r for r in rows if r["condition"] == cond and r["talk"] == "relay" and r["policy"] == "look"}
        rb = {r["seed"]: r for r in rows if r["condition"] == cond and r["talk"] == "seen" and r["policy"] == "look"}
        diffs = [ra[s]["deliveries"] - rb[s]["deliveries"] for s in sorted(ra) if s in rb]
        if diffs:
            mean, lo, hi = boot_ci(diffs)
            comps.append({"condition": cond, "talk": "relay-seen", "a": "look/relay", "b": "look/seen", "mean_diff": mean,
                          "ci95": [lo, hi], "seeds": len(diffs), "a_wins": sum(d > 0 for d in diffs),
                          "b_wins": sum(d < 0 for d in diffs)})
    return {"cells": cells, "comparisons": comps}


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(100, 140)))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = [run_one(s, c, t, p) for c in CONDITIONS for t in TALK for s in seeds for p in POLICIES]
    summary = summarise(rows)
    for c in summary["cells"]:
        extra = f"  leads {c['leads']:.1f} (dropped {c['dead_leads']:.1f})" if "leads" in c else ""
        print(f"{c['condition']:>4} {c['talk']:>5} {c['policy']:>5}: deliveries {c['deliveries']:.2f}{extra}")
    print()
    for c in summary["comparisons"]:
        print(f"{c['condition']:>4} {c['talk']:>10}: {c['a']:>10} - {c['b']:<10} = {c['mean_diff']:+.2f} "
              f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]  wins {c['a_wins']}/{c['seeds']}, losses {c['b_wins']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "foraging_think_alone_check", "seeds": seeds, "summary": summary,
                                     "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
