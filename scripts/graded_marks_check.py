"""Free, fair check of waggle-dance marks ("graded") against think alone in the rumour world. Scripted, $0, no model calls.

Waggle-dance marks: a think-alone ant leaves a stronger mark when it has more of its own evidence (its net count of
private readings, capped at ``--cap``), like a honeybee scout that dances longer for a better site. Liars shout at
the cap. Nothing else changes, and no checker is involved.

A stronger mark is also a louder mark, so a fair test must let think alone be loud too. This check therefore:

1. **Tunes** how much every arm listens to marks (``social_weight``) on development seeds only, picking for each of the
   four arms (fixed or smart fade x alone or graded) the weight with the best mean share right over the ``none``,
   ``burst`` and ``liars`` conditions.
2. **Tests** every tuned arm, plus think alone as it stands (weight 1), on fresh seeds the tuning never saw, at each
   swarm size, under every condition (none, planted, burst, liars), each seeded run paired with its own no-seed
   control.
3. **Compares** on the same seeds: graded vs alone within each fade, and the best graded arm vs the best alone arm
   (best = highest tuning score). Paired percentile bootstrap intervals over seeds.

``right`` is the share of honest ants answering right at the end. ``extra_wrong_mean`` is how many more honest ants
are wrong, on average over the run, than in the same world with nothing planted: the damage done by the seed.

Usage::

    uv run python scripts/graded_marks_check.py --out docs/research/swarm-scale-20261029/graded_marks.json
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
WEIGHTS = {"alone": (1.0, 2.0, 3.0, 4.0, 6.0), "graded": (0.5, 1.0, 1.5, 2.0, 3.0)}
FADES = ("fixed", "smart")


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


def _cfg(base: RumourConfig, n: int, fade: str, weight: float) -> RumourConfig:
    return replace(base, n_ants=n, smart=(fade == "smart"), social_weight=weight)


def run_jobs(jobs: List[Tuple[RumourConfig, int, str, Sequence[str]]], workers: int) -> List[Dict[str, Dict[str, float]]]:
    if workers <= 1:
        return [_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, jobs, chunksize=1))


def tune(base: RumourConfig, n: int, seeds: Sequence[int], workers: int) -> Dict[str, Any]:
    """Pick each arm's listening weight on development seeds. Returns the score of every weight and the pick."""
    keys, jobs = [], []
    for fade in FADES:
        for policy, weights in WEIGHTS.items():
            for w in weights:
                for seed in seeds:
                    keys.append((fade, policy, w, seed))
                    jobs.append((_cfg(base, n, fade, w), seed, policy, TUNE_KINDS))
    results = run_jobs(jobs, workers)
    scores: Dict[Tuple[str, str, float], List[float]] = {}
    for (fade, policy, w, _), res in zip(keys, results):
        scores.setdefault((fade, policy, w), []).extend(res[k]["right"] for k in TUNE_KINDS)
    table = [{"fade": f, "policy": p, "social_weight": w, "score": round(st.mean(v), 4)} for (f, p, w), v in scores.items()]
    picks = {}
    for fade in FADES:
        for policy in WEIGHTS:
            rows = [t for t in table if t["fade"] == fade and t["policy"] == policy]
            best = max(rows, key=lambda t: (t["score"], -t["social_weight"]))  # ties go to the quieter weight
            picks[f"{fade}/{policy}"] = {"social_weight": best["social_weight"], "score": best["score"]}
    return {"n": n, "seeds": list(seeds), "kinds": list(TUNE_KINDS), "table": table, "picks": picks}


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def test(base: RumourConfig, arms: Dict[str, Tuple[str, str, float]], n: int, seeds: Sequence[int],
         workers: int) -> List[Dict[str, Any]]:
    keys, jobs = [], []
    for name, (fade, policy, w) in arms.items():
        for seed in seeds:
            keys.append((name, seed))
            jobs.append((_cfg(base, n, fade, w), seed, policy, KINDS))
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


def comparisons(rows: List[Dict[str, Any]], best_graded: str, best_alone: str) -> List[Dict[str, Any]]:
    pairs = [("fixed/graded*", "fixed/alone*"), ("smart/graded*", "smart/alone*"), (best_graded, best_alone),
             (best_graded, "fixed/alone@1"), (best_graded, "smart/alone@1")]
    out = []
    for n in sorted({r["n"] for r in rows}):
        for a, b in dict.fromkeys(pairs):
            for kind in KINDS:
                out.append(compare(rows, n, kind, a, b, "right"))
                if kind != "none":
                    out.append(compare(rows, n, kind, a, b, "extra_wrong_mean"))
    return out


def print_report(tuning: Dict[str, Any], summary: Dict[str, Any]) -> None:
    print(f"Tuning (n={tuning['n']}, dev seeds {tuning['seeds'][0]}..{tuning['seeds'][-1]}): mean right over "
          f"{', '.join(tuning['kinds'])}")
    for t in tuning["table"]:
        print(f"  {t['fade']:>5}/{t['policy']:<6} weight {t['social_weight']:>4}: {t['score']:.4f}")
    print("  picks:", {k: v["social_weight"] for k, v in tuning["picks"].items()},
          "| best alone:", summary["best_alone"], "| best graded:", summary["best_graded"])
    print()
    print(f"{'n':>5} {'seeded':>8} {'arm':>14} | {'right':>6} | extra wrong")
    for c in summary["cells"]:
        print(f"{c['n']:>5} {c['seeded']:>8} {c['arm']:>14} | {c['right']:>6.4f} | {c['extra_wrong_mean']:.4f}")
    print()
    for c in summary["comparisons"]:
        print(f"n={c['n']:>5} {c['seeded']:>8} {c['metric']:>17}: {c['a']:>14} - {c['b']:<14} = {c['mean_diff']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  higher {c['a_higher']}/{c['b_higher']} of {c['seeds']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--tune-n", type=int, default=256)
    ap.add_argument("--tune-seeds", default="0-9")
    ap.add_argument("--test", default="256:100-119,1024:100-109", help="n:first-last seeds, comma separated")
    ap.add_argument("--accuracy", type=float, default=0.6)
    ap.add_argument("--ticks", type=int, default=60)
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

    tuning = tune(base, a.tune_n, tune_seeds, a.workers)
    arms: Dict[str, Tuple[str, str, float]] = {}
    for key, pick in tuning["picks"].items():
        fade, policy = key.split("/")
        arms[f"{key}*"] = (fade, policy, pick["social_weight"])
    arms["fixed/alone@1"] = ("fixed", "alone", 1.0)
    arms["smart/alone@1"] = ("smart", "alone", 1.0)
    best = {p: max((k for k in tuning["picks"] if k.endswith(p)), key=lambda k: tuning["picks"][k]["score"]) + "*"
            for p in ("/alone", "/graded")}
    rows: List[Dict[str, Any]] = []
    for n, ss in plan:
        rows += test(base, arms, n, ss, a.workers)
    summary = {"best_alone": best["/alone"], "best_graded": best["/graded"], "arms": {k: list(v) for k, v in arms.items()},
               "cells": cells(rows), "comparisons": comparisons(rows, best["/graded"], best["/alone"])}
    print_report(tuning, summary)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        cfg = {k: v for k, v in base.__dict__.items() if k not in ("n_ants", "smart", "social_weight")}
        a.out.write_text(json.dumps({"check": "graded_marks_check", "config": cfg, "tuning": tuning,
                                     "test_plan": [{"n": n, "seeds": ss} for n, ss in plan], "summary": summary,
                                     "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
