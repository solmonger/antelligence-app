"""Free, fair check of the field notebook against the current best in the rumour world. Scripted, $0, no model calls.

Field notebook: each honest ant keeps, for every other author whose mark it has read anywhere, only that author's latest
reading (value, strength as read, tick read), and answers with the faded sum over the whole notebook instead of only the
marks at its current site. Each source counts once, wherever it was met. What the ant says is unchanged (private
evidence only), so nothing heard is relayed. No checker.

Hearing more means a lower weight may be right, so a fair test lets every arm be tuned. This check therefore:

1. **Tunes** on development seeds only, for each of four arms (alone, graded, alone + notebook, graded + notebook; fixed
   fade, ``--cap`` for graded) the listening weight (and, for notebook arms, the notebook half-life, jointly) with the best
   mean share right over the ``none``, ``burst`` and ``liars`` conditions. A pick on the edge of its grid is flagged.
2. **Tests** every tuned arm on fresh seeds the tuning never saw, at each swarm size, under every condition (none,
   planted, burst, liars), each seeded run paired with its own no-seed control.
3. **Compares** on the same seeds: graded + notebook vs graded (the current best), and alone + notebook vs alone.
   Paired percentile bootstrap intervals over seeds.

``right`` is the share of honest ants answering right at the end. ``extra_wrong_mean`` is how many more honest ants are
wrong, on average over the run, than in the same world with nothing planted: the damage done by the seed.

Usage::

    uv run python scripts/notebook_check.py --out docs/research/swarm-scale-20261029/notebook.json
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

from antelligence.worlds.rumour import RumourConfig, RumourWorld

KINDS = ("none", "planted", "burst", "liars")
TUNE_KINDS = ("none", "burst", "liars")
WEIGHTS = {"alone": (1.0, 2.0, 3.0, 4.0, 6.0), "graded": (0.5, 1.0, 1.5, 2.0, 3.0),
           "alone+nb": (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0),
           "graded+nb": (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)}
# Grids were extended twice because the notebook picks sat on an edge (first weights to 2 and half-lives 5 to 20, then
# weights to 4 and half-lives to 80). A half-life of 160 is longer than the 60-tick run, so it means "never fades".
NB_HALF_LIVES = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
ARMS = tuple(WEIGHTS)
# An arm is (policy, weight, notebook half-life); a half-life of 0 means no notebook.
Arm = Tuple[str, float, float]


def _one(job: Tuple[RumourConfig, int, str, Sequence[str]]) -> Dict[str, Dict[str, float]]:
    """Run one arm on one seed under each condition; seeded runs are scored against the no-seed control."""
    cfg, seed, policy, kinds = job
    world = RumourWorld(cfg, seed)
    control = world.run(policy, "none")
    out: Dict[str, Dict[str, float]] = {}
    for kind in kinds:
        r = control if kind == "none" else world.run(policy, kind)
        extra = [w - c for w, c in zip(r.wrong_by_tick, control.wrong_by_tick)]
        out[kind] = {"right": r.right, "wrong": r.wrong, "extra_wrong_mean": round(st.mean(extra), 6)}
    return out


def _cfg(base: RumourConfig, n: int, weight: float, nb_half: float) -> RumourConfig:
    return replace(base, n_ants=n, social_weight=weight, notebook=nb_half > 0, notebook_half_life=nb_half)


def run_jobs(jobs: List[Tuple[RumourConfig, int, str, Sequence[str]]], workers: int) -> List[Dict[str, Dict[str, float]]]:
    if workers <= 1:
        return [_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, jobs, chunksize=1))


def tune(base: RumourConfig, n: int, seeds: Sequence[int], workers: int,
         weights: Dict[str, Sequence[float]] = WEIGHTS, half_lives: Sequence[float] = NB_HALF_LIVES) -> Dict[str, Any]:
    """Pick each arm's listening weight (and notebook half-life) on development seeds. Returns every score and the pick."""
    keys, jobs = [], []
    for arm, ws in weights.items():
        policy = arm.split("+")[0]
        for w in ws:
            for hl in (half_lives if arm.endswith("+nb") else (0.0,)):
                for seed in seeds:
                    keys.append((arm, w, hl, seed))
                    jobs.append((_cfg(base, n, w, hl), seed, policy, TUNE_KINDS))
    results = run_jobs(jobs, workers)
    scores: Dict[Tuple[str, float, float], List[float]] = {}
    for (arm, w, hl, _), res in zip(keys, results):
        scores.setdefault((arm, w, hl), []).extend(res[k]["right"] for k in TUNE_KINDS)
    table = [{"arm": a, "social_weight": w, "notebook_half_life": hl, "score": round(st.mean(v), 4)}
             for (a, w, hl), v in scores.items()]
    picks = {}
    for arm, ws in weights.items():
        rows = [t for t in table if t["arm"] == arm]
        best = max(rows, key=lambda t: (t["score"], -t["social_weight"], -t["notebook_half_life"]))  # ties: quieter
        edge = best["social_weight"] in (min(ws), max(ws))
        picks[arm] = {"social_weight": best["social_weight"], "notebook_half_life": best["notebook_half_life"],
                      "score": best["score"], "weight_on_grid_edge": edge,
                      "half_life_on_grid_edge": arm.endswith("+nb") and best["notebook_half_life"] in (min(half_lives), max(half_lives))}
    return {"n": n, "seeds": list(seeds), "kinds": list(TUNE_KINDS), "table": table, "picks": picks}


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def test(base: RumourConfig, arms: Dict[str, Arm], n: int, seeds: Sequence[int], workers: int) -> List[Dict[str, Any]]:
    keys, jobs = [], []
    for name, (policy, w, hl) in arms.items():
        for seed in seeds:
            keys.append((name, seed))
            jobs.append((_cfg(base, n, w, hl), seed, policy.split("+")[0], KINDS))
    rows = []
    for (name, seed), res in zip(keys, run_jobs(jobs, workers)):
        for kind in KINDS:
            rows.append({"n": n, "seed": seed, "arm": name, "seeded": kind, **res[kind]})
    return rows


def cells(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for n in sorted({r["n"] for r in rows}):
        for kind in KINDS:
            for arm in dict.fromkeys(r["arm"] for r in rows):
                sel = [r for r in rows if r["n"] == n and r["seeded"] == kind and r["arm"] == arm]
                out.append({"n": n, "seeded": kind, "arm": arm, "runs": len(sel),
                            "right": round(st.mean(r["right"] for r in sel), 4),
                            "extra_wrong_mean": round(st.mean(r["extra_wrong_mean"] for r in sel), 4)})
    return out


def compare(rows: List[Dict[str, Any]], n: int, kind: str, a: str, b: str, metric: str) -> Dict[str, Any]:
    ra = {r["seed"]: r[metric] for r in rows if r["n"] == n and r["seeded"] == kind and r["arm"] == a}
    rb = {r["seed"]: r[metric] for r in rows if r["n"] == n and r["seeded"] == kind and r["arm"] == b}
    diffs = [ra[s] - rb[s] for s in sorted(ra) if s in rb]
    mean, lo, hi = boot_ci(diffs)
    return {"n": n, "seeded": kind, "metric": metric, "a": a, "b": b, "mean_diff": mean, "ci95": [lo, hi],
            "seeds": len(diffs), "a_higher": sum(d > 0 for d in diffs), "b_higher": sum(d < 0 for d in diffs)}


def comparisons(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = [("graded+nb", "graded"), ("alone+nb", "alone")]
    out = []
    for n in sorted({r["n"] for r in rows}):
        for a, b in pairs:
            for kind in KINDS:
                out.append(compare(rows, n, kind, a, b, "right"))
                if kind != "none":
                    out.append(compare(rows, n, kind, a, b, "extra_wrong_mean"))
    return out


def print_report(tuning: Dict[str, Any], summary: Dict[str, Any]) -> None:
    print(f"Tuning (n={tuning['n']}, dev seeds {tuning['seeds'][0]}..{tuning['seeds'][-1]}): mean right over "
          f"{', '.join(tuning['kinds'])}")
    for t in tuning["table"]:
        print(f"  {t['arm']:>9} weight {t['social_weight']:>5} nb half-life {t['notebook_half_life']:>4}: {t['score']:.4f}")
    for arm, p in tuning["picks"].items():
        print(f"  pick {arm:>9}: weight {p['social_weight']}, nb half-life {p['notebook_half_life']}, score {p['score']:.4f}"
              f"{'  [WEIGHT ON GRID EDGE]' if p['weight_on_grid_edge'] else ''}"
              f"{'  [HALF-LIFE ON GRID EDGE]' if p['half_life_on_grid_edge'] else ''}")
    print()
    print(f"{'n':>5} {'seeded':>8} {'arm':>10} | {'right':>6} | extra wrong")
    for c in summary["cells"]:
        print(f"{c['n']:>5} {c['seeded']:>8} {c['arm']:>10} | {c['right']:>6.4f} | {c['extra_wrong_mean']:.4f}")
    print()
    for c in summary["comparisons"]:
        print(f"n={c['n']:>5} {c['seeded']:>8} {c['metric']:>17}: {c['a']:>9} - {c['b']:<9} = {c['mean_diff']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  higher {c['a_higher']}/{c['b_higher']} of {c['seeds']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--tune-n", type=int, default=256)
    ap.add_argument("--tune-seeds", default="0-9")
    ap.add_argument("--test", default="256:300-319,1024:300-309", help="n:first-last seeds, comma separated")
    ap.add_argument("--accuracy", type=float, default=0.6)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--weights-nb", default=None, help="comma list replacing the notebook weight grid")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    def seeds(spec: str) -> List[int]:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))

    base = RumourConfig(accuracy=a.accuracy, ticks=a.ticks, grade_cap=a.cap)
    tune_seeds = seeds(a.tune_seeds)
    plan = [(int(n), seeds(s)) for n, s in (part.split(":") for part in a.test.split(","))]
    if set(tune_seeds) & {s for _, ss in plan for s in ss}:
        raise SystemExit("test seeds must be fresh: they overlap the tuning seeds")
    weights = dict(WEIGHTS)
    if a.weights_nb:
        grid = tuple(float(x) for x in a.weights_nb.split(","))
        weights["alone+nb"] = weights["graded+nb"] = grid

    tuning = tune(base, a.tune_n, tune_seeds, a.workers, weights)
    arms: Dict[str, Arm] = {k: (k, p["social_weight"], p["notebook_half_life"]) for k, p in tuning["picks"].items()}
    rows: List[Dict[str, Any]] = []
    for n, ss in plan:
        rows += test(base, arms, n, ss, a.workers)
    summary = {"picks": tuning["picks"], "arms": {k: list(v) for k, v in arms.items()}, "cells": cells(rows),
               "comparisons": comparisons(rows)}
    print_report(tuning, summary)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        cfg = {k: v for k, v in base.__dict__.items() if k not in ("n_ants", "social_weight", "notebook", "notebook_half_life")}
        a.out.write_text(json.dumps({"check": "notebook_check", "config": cfg, "tuning": tuning,
                                     "test_plan": [{"n": n, "seeds": ss} for n, ss in plan], "summary": summary,
                                     "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
