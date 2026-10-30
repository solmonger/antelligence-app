"""Free check: think alone, done included, in the AgentsNet colouring world at swarm size. Scripted, $0, no model calls.

The colouring world is a port of AgentsNet's colouring task (Grötschla et al., arXiv 2507.08616), a task we did not
design. Here it runs at 64 to 1,024 agents under four conditions: ``base``; ``loss`` (each message dropped with
p = 0.3, the pilot's v1); ``liars`` (10% of the agents announce random colours every round, the pilot's v2 at
scale); and ``both``. Policies, all reading the pilot's own sensor:

* ``S``          the pilot's patient greedy: wait for higher-priority neighbours, take one colour, never look again.
* ``S_gate``     S plus the liar probe's earned-trust gate.
* ``S_resend``   the pilot's S' for lost messages (re-announce until acknowledged, guess in the last round).
* ``look``       think alone, then compare, done only when seen done
                 (:class:`antelligence.worlds.graph_coloring.think_alone.LookBeforeDone`).
* ``look_gate``  ``look`` plus the gate.
* ``oracle``     ``look`` told who the liars are (a ceiling, not a policy anyone could run).

Score: share of honest edges properly coloured at the end. Also: honest agents uncoloured, and honest agents that
count themselves done while really clashing. Every policy runs on the same graph, liars and message losses per
seed. Development seeds 100 to 109 only (the pilot's evaluation seeds stay untouched).

Usage::

    uv run python scripts/colouring_think_alone_check.py --ns 256 --out docs/research/swarm-scale-20261029/colouring_n256.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarm_scale_probe import IgnoreSet, pick_liars  # noqa: E402

from antelligence.kernel.field import GridField  # noqa: E402
from antelligence.kernel.scheduler import RunConfig, Scheduler  # noqa: E402
from antelligence.worlds.graph_coloring.graphs import make_graph  # noqa: E402
from antelligence.worlds.graph_coloring.scripts import ResendUntilAcked  # noqa: E402
from antelligence.worlds.graph_coloring.sensing import ColoringSensor  # noqa: E402
from antelligence.worlds.graph_coloring.think_alone import LookBeforeDone, SwarmColouringWorld, done_claims  # noqa: E402
from antelligence.worlds.graph_coloring.world import COLOUR_NAMES  # noqa: E402

CONDITIONS = {"base": (0.0, 0.0), "loss": (0.0, 0.3), "liars": (0.1, 0.0), "both": (0.1, 0.3)}
POLICIES = ("S", "S_gate", "S_resend", "look", "look_gate", "oracle")
FAMILIES = ("ws", "dt")


def make_policy(name: str, seed: int, liars: Sequence[str]):
    sensor = ColoringSensor()
    if name == "S":
        return IgnoreSet(sensor, gate=False)
    if name == "S_gate":
        return IgnoreSet(sensor, gate=True)
    if name == "S_resend":
        return ResendUntilAcked(sensor)
    if name == "look":
        return LookBeforeDone(sensor, seed)
    if name == "look_gate":
        return LookBeforeDone(sensor, seed, gate=True)
    if name == "oracle":
        return LookBeforeDone(sensor, seed, ignore=frozenset(liars))
    raise ValueError(name)


def run_one(n: int, condition: str, family: str, seed: int, policy: str, rounds: int) -> Optional[Dict[str, Any]]:
    if make_graph(family, n, seed).max_degree + 1 > len(COLOUR_NAMES):
        return None  # the sensor names at most 16 colours
    liar_frac, drop_p = CONDITIONS[condition]
    liars = pick_liars(n, liar_frac, seed) if liar_frac else []
    world = SwarmColouringWorld(seed, n_agents=n, rounds=rounds, family=family, liars=liars, drop_p=drop_p)
    pol = make_policy(policy, seed, liars)
    Scheduler(world, {}, GridField(), RunConfig(f"colour-{n}-{seed}", seed=seed, max_ticks=rounds),
              default_policy=pol).run()
    honest = len(world.honest())
    claims = done_claims(world, pol)
    return {"n": n, "condition": condition, "family": family, "seed": seed, "policy": policy, "rounds": rounds,
            "score": round(world.score(), 4),
            "uncoloured_share": round(sum(world.colour[a] is None for a in world.honest()) / honest, 4),
            "done_but_clashing_share": round(claims["claim_done_but_clashing"] / honest, 4),
            "claim_done_share": round(claims["claim_done"] / honest, 4)}


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cells, comps = [], []
    order = list(CONDITIONS)
    keys = sorted({(r["n"], r["condition"], r["rounds"]) for r in rows}, key=lambda k: (k[0], order.index(k[1]), k[2]))
    for n, cond, rounds in keys:
        sel = [r for r in rows if r["n"] == n and r["condition"] == cond and r["rounds"] == rounds]
        for pol in POLICIES:
            p = [r for r in sel if r["policy"] == pol]
            if p:
                cells.append({"n": n, "condition": cond, "rounds": rounds, "policy": pol, "runs": len(p),
                              "score": round(st.mean(r["score"] for r in p), 4),
                              "uncoloured_share": round(st.mean(r["uncoloured_share"] for r in p), 4),
                              "done_but_clashing_share": round(st.mean(r["done_but_clashing_share"] for r in p), 4)})
        for a, b in (("look", "S"), ("look_gate", "S_gate"), ("look", "S_resend"), ("look_gate", "look")):
            ra = {(r["family"], r["seed"]): r for r in sel if r["policy"] == a}
            rb = {(r["family"], r["seed"]): r for r in sel if r["policy"] == b}
            diffs = [ra[k]["score"] - rb[k]["score"] for k in sorted(ra) if k in rb]
            if diffs:
                mean, lo, hi = boot_ci(diffs)
                comps.append({"n": n, "condition": cond, "rounds": rounds, "a": a, "b": b, "mean_diff": mean,
                              "ci95": [lo, hi], "pairs": len(diffs), "a_wins": sum(d > 0 for d in diffs),
                              "b_wins": sum(d < 0 for d in diffs)})
    return {"cells": cells, "comparisons": comps}


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", default="64,256")
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(100, 110)))
    ap.add_argument("--rounds", default="8")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = []
    for n in (int(x) for x in a.ns.split(",")):
        for rounds in (int(x) for x in a.rounds.split(",")):
            for cond in a.conditions.split(","):
                for fam in FAMILIES:
                    for seed in (int(x) for x in a.seeds.split(",")):
                        for pol in POLICIES:
                            r = run_one(n, cond, fam, seed, pol, rounds)
                            if r:
                                rows.append(r)
    summary = summarise(rows)
    print(f"{'n':>5} {'cond':>6} {'rnd':>3} {'policy':>10} | {'score':>6} {'uncol':>6} {'done&clash':>10}")
    for c in summary["cells"]:
        print(f"{c['n']:>5} {c['condition']:>6} {c['rounds']:>3} {c['policy']:>10} | {c['score']:>6.3f} "
              f"{c['uncoloured_share']:>6.3f} {c['done_but_clashing_share']:>10.3f}")
    print()
    for c in summary["comparisons"]:
        print(f"n={c['n']:>5} {c['condition']:>6} r={c['rounds']:>2}: {c['a']:>9} - {c['b']:<8} = {c['mean_diff']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  wins {c['a_wins']}/{c['pairs']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "colouring_think_alone_check", "conditions": CONDITIONS,
                                     "summary": summary, "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
