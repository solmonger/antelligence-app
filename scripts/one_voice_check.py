"""Free, fair check of two notebook fixes against the current best in the rumour world. Scripted, $0, no model calls.

The current best (draft PR 24) is waggle-dance marks plus a field notebook. Its cost: a wrong mark planted as strong as
eight ants is remembered at that strength for most of the run, so a burst of misinformation does about four times the
harm. Two fixes, each an option on ``RumourConfig``, off by default:

* ``notebook_one_voice``: a remembered source counts for at most what one ant can say (the waggle cap).
* ``notebook_consolidate``: a source seen writing only once fades fast; seen writing at two ticks, it fades slowly.

Every arm (the current best included) has its listening weight, notebook half-life and, where used, its short half-life
tuned on development seeds 0 to 9 at 256 ants (mean right over none, burst and liars). Then each tuned arm runs on fresh
seeds the tuning never saw, at 256 and 1,024 ants, under none, planted, burst and liars, each paired with its own
no-seed control. Each fix is compared with the current best on the same seeds (paired bootstrap over seeds).

Usage::

    uv run python scripts/one_voice_check.py --workers 4 --out docs/research/swarm-scale-20261029/one_voice.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from antelligence.worlds.rumour import RumourConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebook_check import KINDS, TUNE_KINDS, boot_ci, cells, run_jobs  # noqa: E402

WEIGHTS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
NB_HALF_LIVES = (20.0, 40.0, 80.0, 160.0)
SHORT_HALF_LIVES = (2.0, 5.0, 10.0, 20.0)
# arm -> (one voice, consolidate). All arms are graded (waggle cap 3) + notebook, fixed fade.
ARMS = {"best": (False, False), "voice": (True, False), "consol": (False, True), "voice+consol": (True, True)}
Pick = Tuple[float, float, float]  # weight, notebook half-life, short half-life (0 = off)


def _cfg(base: RumourConfig, n: int, arm: str, pick: Pick) -> RumourConfig:
    voice, _ = ARMS[arm]
    w, hl, short = pick
    return replace(base, n_ants=n, social_weight=w, notebook=True, notebook_half_life=hl,
                   notebook_one_voice=voice, notebook_consolidate=short)


def grid(arm: str) -> List[Pick]:
    shorts = SHORT_HALF_LIVES if ARMS[arm][1] else (0.0,)
    return [(w, hl, s) for w in WEIGHTS for hl in NB_HALF_LIVES for s in shorts if not s or s < hl]


def tune(base: RumourConfig, n: int, seeds: Sequence[int], workers: int) -> Dict[str, Any]:
    keys, jobs = [], []
    for arm in ARMS:
        for pick in grid(arm):
            for seed in seeds:
                keys.append((arm, pick))
                jobs.append((_cfg(base, n, arm, pick), seed, "graded", TUNE_KINDS))
    scores: Dict[Tuple[str, Pick], List[float]] = {}
    for key, res in zip(keys, run_jobs(jobs, workers)):
        scores.setdefault(key, []).extend(res[k]["right"] for k in TUNE_KINDS)
    table = [{"arm": a, "social_weight": p[0], "notebook_half_life": p[1], "short_half_life": p[2],
              "score": round(st.mean(v), 4)} for (a, p), v in scores.items()]
    picks = {}
    for arm in ARMS:
        rows = [t for t in table if t["arm"] == arm]
        b = max(rows, key=lambda t: (t["score"], -t["social_weight"], -t["notebook_half_life"], t["short_half_life"]))
        edges = []
        if b["social_weight"] in (min(WEIGHTS), max(WEIGHTS)):
            edges.append("weight")
        if b["notebook_half_life"] in (min(NB_HALF_LIVES), max(NB_HALF_LIVES)):
            edges.append("notebook half-life")
        if ARMS[arm][1] and b["short_half_life"] in (min(SHORT_HALF_LIVES), max(SHORT_HALF_LIVES)):
            edges.append("short half-life")
        picks[arm] = {**{k: b[k] for k in ("social_weight", "notebook_half_life", "short_half_life", "score")},
                      "on_grid_edge": edges}
    return {"n": n, "seeds": list(seeds), "kinds": list(TUNE_KINDS), "table": table, "picks": picks}


def test(base: RumourConfig, picks: Dict[str, Pick], n: int, seeds: Sequence[int], workers: int) -> List[Dict[str, Any]]:
    keys, jobs = [], []
    for arm, pick in picks.items():
        for seed in seeds:
            keys.append((arm, seed))
            jobs.append((_cfg(base, n, arm, pick), seed, "graded", KINDS))
    return [{"n": n, "seed": seed, "arm": arm, "seeded": kind, **res[kind]}
            for (arm, seed), res in zip(keys, run_jobs(jobs, workers)) for kind in KINDS]


def compare(rows: List[Dict[str, Any]], n: int, kind: str, a: str, b: str, metric: str) -> Dict[str, Any]:
    ra = {r["seed"]: r[metric] for r in rows if (r["n"], r["seeded"], r["arm"]) == (n, kind, a)}
    rb = {r["seed"]: r[metric] for r in rows if (r["n"], r["seeded"], r["arm"]) == (n, kind, b)}
    diffs = [ra[s] - rb[s] for s in sorted(ra) if s in rb]
    mean, lo, hi = boot_ci(diffs)
    return {"n": n, "seeded": kind, "metric": metric, "a": a, "b": b, "mean_diff": mean, "ci95": [lo, hi],
            "seeds": len(diffs), "a_higher": sum(d > 0 for d in diffs), "b_higher": sum(d < 0 for d in diffs)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tune-n", type=int, default=256)
    ap.add_argument("--tune-seeds", default="0-9")
    ap.add_argument("--test", default="256:400-419,1024:400-409", help="n:first-last seeds, comma separated")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    def seeds(spec: str) -> List[int]:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))

    base = RumourConfig(grade_cap=3.0)
    tune_seeds = seeds(a.tune_seeds)
    plan = [(int(n), seeds(s)) for n, s in (part.split(":") for part in a.test.split(","))]
    if set(tune_seeds) & {s for _, ss in plan for s in ss}:
        raise SystemExit("test seeds must be fresh: they overlap the tuning seeds")

    tuning = tune(base, a.tune_n, tune_seeds, a.workers)
    picks = {k: (p["social_weight"], p["notebook_half_life"], p["short_half_life"]) for k, p in tuning["picks"].items()}
    for arm, p in tuning["picks"].items():
        print(f"pick {arm:>13}: weight {p['social_weight']}, notebook half-life {p['notebook_half_life']}, short "
              f"half-life {p['short_half_life']}, dev score {p['score']:.4f}  edge: {p['on_grid_edge'] or 'none'}")
    rows: List[Dict[str, Any]] = []
    for n, ss in plan:
        rows += test(base, picks, n, ss, a.workers)
    comps = [compare(rows, n, kind, arm, "best", m) for n, _ in plan for arm in ARMS if arm != "best"
             for kind in KINDS for m in (("right",) if kind == "none" else ("right", "extra_wrong_mean"))]
    summary = {"picks": tuning["picks"], "cells": cells(rows), "comparisons": comps}
    print(f"\n{'n':>5} {'seeded':>8} {'arm':>13} | {'right':>6} | extra wrong")
    for c in summary["cells"]:
        print(f"{c['n']:>5} {c['seeded']:>8} {c['arm']:>13} | {c['right']:>6.4f} | {c['extra_wrong_mean']:.4f}")
    print()
    for c in comps:
        print(f"n={c['n']:>5} {c['seeded']:>8} {c['metric']:>17}: {c['a']:>13} - best = {c['mean_diff']:+.4f} "
              f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  higher {c['a_higher']}/{c['b_higher']} of {c['seeds']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"check": "one_voice_check", "tuning": tuning, "summary": summary, "rows": rows,
                                     "test_plan": [{"n": n, "seeds": ss} for n, ss in plan]},
                                    indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
