"""Free probe: does the field notebook still pay when a burst of wrong marks comes from many authors? Scripted, $0.

In the rumour world every planted mark carries one author name, so a notebook counts a whole burst once, however many
sites it meets it at. A burst by many distinct authors (one per site) is the harder case: each copy then counts. This
probe renames each planted mark to its own author and re-runs the tuned arms from ``notebook.json`` on the same fresh
seeds, paired by seed, scored as in ``notebook_check.py`` (share right at the end; extra honest ants wrong over the run
against the same world with nothing planted).

``--one-voice`` turns on ``notebook_one_voice`` for the notebook arms (each remembered source counts as at most one ant's
strongest mark), to see whether it also cuts the harm of a burst from many authors.

Usage::

    uv run python scripts/notebook_burst_probe.py --out docs/research/swarm-scale-20261029/notebook_burst_probe.json
    uv run python scripts/notebook_burst_probe.py --one-voice \
        --baseline docs/research/swarm-scale-20261029/notebook_burst_probe.json \
        --out docs/research/swarm-scale-20261029/one_voice_burst_probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from antelligence.kernel import fading
from antelligence.worlds.rumour import RumourConfig, RumourWorld
from antelligence.worlds.rumour.world import PLANTED

REPO = Path(__file__).resolve().parents[1]
PAIRS = (("graded+nb", "graded"), ("alone+nb", "alone"))


def _many_authors() -> None:
    """Give each planted mark its own author (``planted@<site>``), so a notebook counts each copy."""
    real = fading.FadingMemory.deposit
    if getattr(real, "_many_authors", False):
        return

    def deposit(self, place, author, *args, **kwargs):
        return real(self, place, f"{PLANTED}@{place}" if author == PLANTED else author, *args, **kwargs)

    deposit._many_authors = True
    fading.FadingMemory.deposit = deposit


def _one(job: Tuple[RumourConfig, int, str, bool]) -> Dict[str, float]:
    cfg, seed, policy, many = job
    if many:
        _many_authors()
    world = RumourWorld(cfg, seed)
    control = world.run(policy, "none")
    r = world.run(policy, "burst")
    return {"right": r.right, "extra_wrong_mean": st.mean(w - c for w, c in zip(r.wrong_by_tick, control.wrong_by_tick))}


def boot_ci(diffs: Sequence[float], reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    import random
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(reps))
    return round(st.mean(diffs), 4), round(means[int(0.025 * reps)], 4), round(means[int(0.975 * reps) - 1], 4)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--picks", type=Path, default=REPO / "docs/research/swarm-scale-20261029/notebook.json")
    ap.add_argument("--one-voice", action="store_true", help="notebook arms remember each source as one ant's word")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="an earlier probe JSON: pair each notebook arm here against the same arm there, by seed")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rec = json.loads(a.picks.read_text())
    picks = rec["summary"]["picks"]
    base = RumourConfig(**{k: v for k, v in rec["config"].items() if k in RumourConfig.__dataclass_fields__})
    plan = [(p["n"], p["seeds"]) for p in rec["test_plan"]]
    jobs, keys = [], []
    for n, seeds in plan:
        for arm, pk in picks.items():
            cfg = replace(base, n_ants=n, social_weight=pk["social_weight"], notebook=pk["notebook_half_life"] > 0,
                          notebook_half_life=pk["notebook_half_life"],
                          notebook_one_voice=a.one_voice and pk["notebook_half_life"] > 0)
            for seed in seeds:
                for many in (False, True):
                    jobs.append((cfg, seed, arm.split("+")[0], many))
                    keys.append((n, arm, seed, many))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(_one, [j for j in jobs if not j[3]], chunksize=4))
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_many_authors) as pool:
        res_many = list(pool.map(_one, [j for j in jobs if j[3]], chunksize=4))
    it_one, it_many = iter(res), iter(res_many)
    rows = [{"n": n, "arm": arm, "seed": seed, "many_authors": many, **(next(it_many) if many else next(it_one))}
            for (n, arm, seed, many) in keys]
    cells, comps = [], []
    for n, seeds in plan:
        for many in (False, True):
            for arm in picks:
                sel = [r for r in rows if (r["n"], r["arm"], r["many_authors"]) == (n, arm, many)]
                cells.append({"n": n, "many_authors": many, "arm": arm,
                              **{m: round(st.mean(r[m] for r in sel), 4) for m in ("right", "extra_wrong_mean")}})
            for x, y in PAIRS:
                for m in ("right", "extra_wrong_mean"):
                    rx = {r["seed"]: r[m] for r in rows if (r["n"], r["arm"], r["many_authors"]) == (n, x, many)}
                    ry = {r["seed"]: r[m] for r in rows if (r["n"], r["arm"], r["many_authors"]) == (n, y, many)}
                    diffs = [rx[s] - ry[s] for s in seeds]
                    mean, lo, hi = boot_ci(diffs)
                    comps.append({"n": n, "many_authors": many, "a": x, "b": y, "metric": m, "mean_diff": mean,
                                  "ci95": [lo, hi], "a_higher": sum(d > 0 for d in diffs), "seeds": len(diffs)})
    if a.baseline:
        base_rows = json.loads(a.baseline.read_text())["rows"]
        for n, seeds in plan:
            for many in (False, True):
                for arm in (x for x in picks if x.endswith("+nb")):
                    for m in ("right", "extra_wrong_mean"):
                        key = lambda r: (r["n"], r["arm"], r["many_authors"]) == (n, arm, many)
                        rx = {r["seed"]: r[m] for r in rows if key(r)}
                        ry = {r["seed"]: r[m] for r in base_rows if key(r)}
                        diffs = [rx[s] - ry[s] for s in seeds]
                        mean, lo, hi = boot_ci(diffs)
                        comps.append({"n": n, "many_authors": many, "a": arm, "b": f"{arm} (baseline)", "metric": m,
                                      "mean_diff": mean, "ci95": [lo, hi], "a_higher": sum(d > 0 for d in diffs),
                                      "a_lower": sum(d < 0 for d in diffs), "seeds": len(diffs)})
    for c in cells:
        print(f"  n={c['n']:>5} {'many authors' if c['many_authors'] else 'one author  '} {c['arm']:>10}: "
              f"right {c['right']:.4f}  extra wrong {c['extra_wrong_mean']:.4f}")
    for c in comps:
        print(f"  n={c['n']:>5} {'many authors' if c['many_authors'] else 'one author  '} {c['metric']:>16}: "
              f"{c['a']} - {c['b']} = {c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] "
              f"higher {c['a_higher']}/{c['seeds']}")
    if a.out:
        a.out.write_text(json.dumps({"check": "notebook_burst_probe", "one_voice": a.one_voice, "cells": cells, "comparisons": comps,
                                     "rows": rows}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
