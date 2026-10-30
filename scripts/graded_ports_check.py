"""Free check: waggle-dance marks ported to the colouring and foraging worlds, against think alone. Scripted, $0.

Waggle-dance marks (the rumour world's ``graded`` policy, see ``docs/research/swarm-scale-20261029/GRADED_MARKS.md``):
a signal says how sure its sender is, and listeners lean on surer signals. Cap 2 (the safer cap from the rumour world).
Both worlds are tasks we did not design; neither has a listening weight to tune, so the comparison is direct.

* **Colouring** (AgentsNet port): ``GradedLook`` sends with its colour how sure it is (1 plus its clean looks in a row,
  capped). On a clash the less sure ant moves; equally sure ants fall back to think alone's rule. Liars claim the cap.
  Against ``look`` (think alone) in the same :class:`GradedColouringWorld`, under base, lost messages, 10% liars, and
  both, on graph families ws and dt. Fewer rounds than the earlier check, because at 8 rounds think alone is already
  at the ceiling (see ``colouring_n256.json``), so only speed can differ.
* **Foraging** (SwarmBench port): with relayed talk on, a first-hand "food is north of me" carries strength 2 and a
  relayed memory strength 1; ``GradedForager`` follows the surest lead, nearest first among equals. ``shout`` is the
  liar case: relayed (possibly stale) memories claim strength 2. Against ``look`` (``LookForager``).

Seeds 200 and up: fresh for both worlds (earlier checks used 100 to 139). Paired by seed, bootstrap intervals.

Usage::

    uv run python scripts/graded_ports_check.py --out docs/research/swarm-scale-20261029/graded_ports.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarm_scale_probe import pick_liars  # noqa: E402

from antelligence.kernel.field import GridField  # noqa: E402
from antelligence.kernel.scheduler import RunConfig, Scheduler  # noqa: E402
from antelligence.worlds.foraging_sb.think_alone import GradedForager, LookForager, RelayForagingWorld  # noqa: E402
from antelligence.worlds.foraging_sb.world import STEPS, V1_PICKUPS, V2_FLIP_TICK  # noqa: E402
from antelligence.worlds.graph_coloring.graphs import make_graph  # noqa: E402
from antelligence.worlds.graph_coloring.sensing import ColoringSensor  # noqa: E402
from antelligence.worlds.graph_coloring.think_alone import GradedColouringWorld, GradedLook, LookBeforeDone  # noqa: E402
from antelligence.worlds.graph_coloring.think_alone import done_claims  # noqa: E402
from antelligence.worlds.graph_coloring.world import COLOUR_NAMES  # noqa: E402

COLOUR_CONDITIONS = {"base": (0.0, 0.0), "loss": (0.0, 0.3), "liars": (0.1, 0.0), "both": (0.1, 0.3)}
FAMILIES = ("ws", "dt")
FORAGE_CONDITIONS = ("base", "v1", "v2")
TALK = ("seen", "relay", "shout")


def colour_one(job: Tuple[int, str, str, int, str, int, int]) -> Optional[Dict[str, Any]]:
    n, condition, family, seed, policy, rounds, cap = job
    if make_graph(family, n, seed).max_degree + 1 > len(COLOUR_NAMES):
        return None
    liar_frac, drop_p = COLOUR_CONDITIONS[condition]
    liars = pick_liars(n, liar_frac, seed) if liar_frac else []
    world = GradedColouringWorld(seed, n_agents=n, rounds=rounds, family=family, liars=liars, drop_p=drop_p, cap=cap)
    pol = GradedLook(ColoringSensor(), seed, cap=cap) if policy == "graded" else LookBeforeDone(ColoringSensor(), seed)
    Scheduler(world, {}, GridField(), RunConfig(f"colour-{n}-{seed}", seed=seed, max_ticks=rounds),
              default_policy=pol).run()
    honest = len(world.honest())
    claims = done_claims(world, pol)
    return {"world": "colouring", "n": n, "condition": condition, "family": family, "seed": seed, "policy": policy,
            "rounds": rounds, "score": round(world.score(), 6),
            "done_but_clashing_share": round(claims["claim_done_but_clashing"] / honest, 6)}


def forage_one(job: Tuple[int, str, str, str, int]) -> Dict[str, Any]:
    seed, condition, talk, policy, cap = job
    world = RelayForagingWorld(seed, condition, relay=(talk != "seen"), shout=(talk == "shout"), cap=cap,
                               n_pickups=V1_PICKUPS, flip_tick=V2_FLIP_TICK, max_steps=STEPS)
    pol = GradedForager() if policy == "graded" else LookForager()
    Scheduler(world, {}, GridField(), RunConfig(f"forage-{condition}-{seed}", seed=seed, max_ticks=STEPS),
              default_policy=pol, record_frames=False).run()
    return {"world": "foraging", "condition": condition, "talk": talk, "seed": seed, "policy": policy,
            "score": world.deliveries, **pol.lead_stats()}


def _pool_map(fn, jobs: List[Any], workers: int) -> List[Any]:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs, chunksize=4))


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def summarise(rows: List[Dict[str, Any]], cell_keys: Sequence[str], pair_keys: Sequence[str],
              extra: Sequence[str] = ()) -> Dict[str, Any]:
    cells, comps = [], []
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in cell_keys), []).append(r)
    for key, sel in groups.items():
        cell = dict(zip(cell_keys, key))
        for pol in ("look", "graded"):
            p = [r for r in sel if r["policy"] == pol]
            cells.append({**cell, "policy": pol, "runs": len(p), "score": round(st.mean(r["score"] for r in p), 4),
                          **{m: round(st.mean(r[m] for r in p), 4) for m in extra}})
        ra = {tuple(r[k] for k in pair_keys): r["score"] for r in sel if r["policy"] == "graded"}
        rb = {tuple(r[k] for k in pair_keys): r["score"] for r in sel if r["policy"] == "look"}
        diffs = [ra[k] - rb[k] for k in sorted(ra) if k in rb]
        mean, lo, hi = boot_ci(diffs)
        comps.append({**cell, "a": "graded", "b": "look", "mean_diff": mean, "ci95": [lo, hi], "pairs": len(diffs),
                      "a_higher": sum(d > 0 for d in diffs), "b_higher": sum(d < 0 for d in diffs)})
    return {"cells": cells, "comparisons": comps}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--ns", default="256,1024")
    ap.add_argument("--rounds", default="3,4")
    ap.add_argument("--colour-seeds", default="200-219")
    ap.add_argument("--forage-seeds", default="200-279")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    def seeds(spec: str) -> List[int]:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))

    cjobs = [(n, cond, fam, seed, pol, rounds, a.cap) for n in (int(x) for x in a.ns.split(","))
             for rounds in (int(x) for x in a.rounds.split(",")) for cond in COLOUR_CONDITIONS for fam in FAMILIES
             for seed in seeds(a.colour_seeds) for pol in ("look", "graded")]
    crows = [r for r in _pool_map(colour_one, cjobs, a.workers) if r]
    fjobs = [(seed, cond, talk, pol, a.cap) for cond in FORAGE_CONDITIONS for talk in TALK
             for seed in seeds(a.forage_seeds) for pol in ("look", "graded")]
    frows = _pool_map(forage_one, fjobs, a.workers)
    colouring = summarise(crows, ("n", "rounds", "condition"), ("family", "seed"), ("done_but_clashing_share",))
    foraging = summarise(frows, ("condition", "talk"), ("seed",), ("leads", "dead_leads"))

    print("Colouring: share of honest edges properly coloured (graded minus look, 95% CI, pairs higher/lower)")
    for c in colouring["comparisons"]:
        cell = {x["policy"]: x for x in colouring["cells"]
                if (x["n"], x["rounds"], x["condition"]) == (c["n"], c["rounds"], c["condition"])}
        print(f"  n={c['n']:>5} r={c['rounds']} {c['condition']:>6}: look {cell['look']['score']:.4f} "
              f"graded {cell['graded']['score']:.4f}  diff {c['mean_diff']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  {c['a_higher']}/{c['b_higher']} of {c['pairs']}")
    print("Foraging: deliveries")
    for c in foraging["comparisons"]:
        cell = {x["policy"]: x for x in foraging["cells"] if (x["condition"], x["talk"]) == (c["condition"], c["talk"])}
        print(f"  {c['condition']:>4} {c['talk']:>5}: look {cell['look']['score']:.2f} graded {cell['graded']['score']:.2f}"
              f"  diff {c['mean_diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  "
              f"{c['a_higher']}/{c['b_higher']} of {c['pairs']}  dead leads {cell['look']['dead_leads']:.2f} -> "
              f"{cell['graded']['dead_leads']:.2f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "graded_ports_check", "cap": a.cap, "colour_conditions": COLOUR_CONDITIONS,
                                     "colouring": colouring, "foraging": foraging, "rows": crows + frows},
                                    indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
