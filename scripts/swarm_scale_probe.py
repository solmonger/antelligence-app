"""Swarm-at-scale probe: many liars, many agents, one model-free gate. No model calls, no network, $0.

Groundwork check for direction D1 in ``docs/plans/2026-10-29-swarm-at-scale.md`` ("trusted gate, untrusted
swarm"). It reuses the v4 pilot colouring world and its scripts, and changes only two things:

* **Many liars.** The pilot's v2 variant has one fixed liar (agent "1"). Here a seeded fraction ``f`` of the
  agents are liars. A liar announces a random colour to each neighbour every round. Only edges between two
  honest agents are scored, exactly as in v2.
* **Many agents.** ``n`` from 8 up to thousands, on the Watts-Strogatz and Delaunay families (their palettes
  stay within the 16 colour names the sensor can render). Barabasi-Albert hubs exceed that, so it is left out.

Three honest policies, all reading the same ``SensedState`` as every pilot arm:

* ``S``      the pilot's patient greedy (waits for every lower-id neighbour to settle; never guesses).
* ``gate``   S plus one model-free admission rule: a neighbour whose announced colour has changed
             ``GATE_FLIPS`` or more times is no longer heard (never waited for, its colour not avoided).
             Honest agents in this protocol commit to a colour once, so a report that keeps changing breaks a
             protocol invariant. The gate reads behaviour only; it is never told who lies.
* ``oracle`` S' generalised: told every liar's id and ignores them (the ceiling, as in the pilot).

Output: one JSON line per (n, f, family, seed, policy) with score, uncoloured honest agents and seconds per
agent-decision, then a summary table. Usage::

    uv run python scripts/swarm_scale_probe.py --out docs/research/swarm-scale-20261029/probe.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from antelligence.kernel.field import GridField
from antelligence.kernel.scheduler import RunConfig, Scheduler
from antelligence.kernel.types import NOOP, Intent, LocalView
from antelligence.worlds.graph_coloring.graphs import make_graph
from antelligence.worlds.graph_coloring.scripts import PatientGreedy
from antelligence.worlds.graph_coloring.sensing import ColoringSensor
from antelligence.worlds.graph_coloring.world import COLOUR_NAMES, GraphColoringWorld, colour_action

GATE_FLIPS = 2
FAMILIES = ("ws", "dt")
POLICIES = ("S", "gate", "oracle")


class ManyLiarsWorld(GraphColoringWorld):
    """The pilot colouring world with a set of liars instead of one. ``base`` condition otherwise."""

    def __init__(self, seed: int, *, n_agents: int, rounds: int, family: str, liars: Iterable[str]) -> None:
        super().__init__(seed, "base", n_agents=n_agents, rounds=rounds, family=family)
        self.liars: FrozenSet[str] = frozenset(liars)

    def step_environment(self, tick: int) -> None:
        outgoing = []
        for u in self.agents():
            liar = u in self.liars
            if u not in self._announcing and not liar:
                continue
            for v in self._adj[u]:
                announced = self._lie_rng.randrange(self.palette) if liar else self.colour[u]
                if liar:
                    self.counters["lied"] += 1
                if announced is None:
                    continue
                prior = self.known[u].get(v)
                outgoing.append((u, v, announced, prior["colour"] if prior else None))
        self._announcing = set()
        self.fresh = {a: set() for a in self._adj}
        for u, v, announced, ack in outgoing:
            self.counters["sent"] += 1
            old = self.known[v].get(u)
            self.known[v][u] = {"colour": announced, "ack": ack, "round": tick,
                                "prev": old["colour"] if old else None}
            self.fresh[v].add(u)
        self._revision += 1

    def scored_edges(self):
        return [(str(u), str(v)) for u, v in self.graph.edges
                if str(u) not in self.liars and str(v) not in self.liars]

    def uncoloured_honest(self) -> int:
        return sum(self.colour[a] is None for a in self.agents() if a not in self.liars)


class IgnoreSet(PatientGreedy):
    """Patient greedy that does not hear a set of neighbours. Fixed set = oracle; growing set = gate."""

    def __init__(self, sensor: ColoringSensor, ignore: FrozenSet[str] = frozenset(), *, gate: bool) -> None:
        super().__init__(sensor)
        self.fixed_ignore, self.gate = ignore, gate
        self.flips: Dict[str, Dict[str, int]] = {}  # receiver -> sender -> observed changes

    def decide(self, view: LocalView) -> Intent:
        f = self._sensor.sense(view).fields
        me = f["agent"]
        if self.gate:
            seen = self.flips.setdefault(me, {})
            for u in f["flipped"]:
                seen[u] = seen.get(u, 0) + 1
            ignored = {u for u, k in seen.items() if k >= GATE_FLIPS}
        else:
            ignored = set(self.fixed_ignore)
        f = dict(f, neighbours=[u for u in f["neighbours"] if u not in ignored],
                 flipped=[u for u in f["flipped"] if u not in ignored])
        choice = self.choose(f)
        return Intent(NOOP) if choice is None else Intent(colour_action(choice))


def pick_liars(n: int, frac: float, seed: int) -> List[str]:
    k = round(n * frac)
    return sorted(random.Random(f"liars:{n}:{frac}:{seed}").sample([str(i) for i in range(n)], k), key=int)


def run_one(n: int, frac: float, family: str, seed: int, policy: str, rounds: int) -> Optional[Dict[str, Any]]:
    graph = make_graph(family, n, seed)
    if graph.max_degree + 1 > len(COLOUR_NAMES):
        return None  # the sensor names at most 16 colours
    liars = pick_liars(n, frac, seed)
    world = ManyLiarsWorld(seed, n_agents=n, rounds=rounds, family=family, liars=liars)
    sensor = ColoringSensor()
    if policy == "S":
        pol = IgnoreSet(sensor, gate=False)
    elif policy == "gate":
        pol = IgnoreSet(sensor, gate=True)
    else:
        pol = IgnoreSet(sensor, frozenset(liars), gate=False)
    t0 = time.perf_counter()
    Scheduler(world, {}, GridField(), RunConfig(f"probe-{n}-{seed}", seed=seed, max_ticks=rounds),
              default_policy=pol).run()
    wall = time.perf_counter() - t0
    honest = n - len(liars)
    return {"n": n, "liar_frac": frac, "liars": len(liars), "family": family, "seed": seed, "policy": policy,
            "rounds": rounds, "score": round(world.score(), 4),
            "uncoloured_honest_share": round(world.uncoloured_honest() / honest, 4) if honest else None,
            "s_per_agent_decision": wall / (n * rounds)}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", default="8,64,512,2048")
    ap.add_argument("--fracs", default="0,0.02,0.1,0.3")
    ap.add_argument("--seeds", default="100,101,102,103,104")  # the pilot's development seeds only
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    ns = [int(x) for x in a.ns.split(",")]
    fracs = [float(x) for x in a.fracs.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []
    for n in ns:
        for frac in fracs:
            for family in FAMILIES:
                for seed in seeds:
                    for policy in POLICIES:
                        r = run_one(n, frac, family, seed, policy, a.rounds)
                        if r is not None:
                            rows.append(r)
    summary = []
    print(f"{'n':>5} {'liars':>6} | {'S':>6} {'gate':>6} {'oracle':>6} | {'S stuck':>7} {'gate stuck':>10}"
          f" | {'ms/decision':>11}")
    for n in ns:
        for frac in fracs:
            cell = {p: [r for r in rows if r["n"] == n and r["liar_frac"] == frac and r["policy"] == p]
                    for p in POLICIES}
            if not cell["S"]:
                continue
            m = {p: mean(r["score"] for r in cell[p]) for p in POLICIES}
            stuck = {p: mean(r["uncoloured_honest_share"] for r in cell[p]) for p in ("S", "gate")}
            ms = 1000 * mean(r["s_per_agent_decision"] for r in rows if r["n"] == n)
            summary.append({"n": n, "liar_frac": frac, "runs_per_policy": len(cell["S"]),
                            "mean_score": {p: round(v, 4) for p, v in m.items()},
                            "mean_uncoloured_honest_share": {p: round(v, 4) for p, v in stuck.items()},
                            "ms_per_agent_decision": round(ms, 3)})
            print(f"{n:>5} {frac:>6.2f} | {m['S']:>6.3f} {m['gate']:>6.3f} {m['oracle']:>6.3f} | "
                  f"{stuck['S']:>7.3f} {stuck['gate']:>10.3f} | {ms:>11.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"probe": "swarm_scale_probe", "gate_flips": GATE_FLIPS,
                                     "rounds": a.rounds, "families": list(FAMILIES), "seeds": seeds,
                                     "summary": summary, "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
