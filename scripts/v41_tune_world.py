#!/usr/bin/env python3
"""Tune config/v4_1/world.yaml total_drug and dose_threshold (D_t) with scripts S and S' on DEV cases only.

Cases come from the L1 store through the tumor_mri store adapter (data/v4_1/cases/<case>/), so the world sees the real
perception arrays, the stored knowledge regions and the true outside-crop truth count. Dev = dev_a + dev_b of
config/v4_1/split.json; eval and reserve cases are refused before anything is loaded.

Targets (goal section 4.2): median S' coverage 0.70-0.80, median S 0.30-0.50, S' - S >= 0.15 on >= 80% of cases.
If no grid point meets them a widened band is tried (S' 0.65-0.85 / S 0.25-0.55 / gap on >= 70%; the status becomes
`fallback_widened_band`). If that fails too, the point whose medians sit in the widened bands with the most cases showing
the gap is reported (`fallback_gap_unmet`), else the point nearest the band centres (`no_feasible_point`); world.yaml is
written in every case with a comment saying which.

Coverage per case is the mean over --seeds (default 1,2). Dose is linear in the drug, so only D_t / total_drug changes
behaviour: the search is one-dimensional in that ratio (several D_t values are accepted and share episodes). Stage 1 scans
a wide geometric grid, stage 2 refines around the best point.

  python scripts/v41_tune_world.py --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker process: the pool is the parallelism
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antelligence.worlds.tumor_mri.config import WorldConfig  # noqa: E402
from antelligence.worlds.tumor_mri.tuning import (STRICT, WIDENED, SplitError, assert_dev_only,  # noqa: E402
                                                  episode_coverages_for_case, grid_search, init_worker, load_split,
                                                  timed_episode_coverages_for_case)

HEADER = """# v4.1 tumor_mri world constants.
# total_drug and dose_threshold were tuned by scripts/v41_tune_world.py on the 40 DEV cases (dev_a + dev_b) with scripts
# S and S' only, seeds {seeds}, through the L1 store adapter. Status: {status} ({band} band).
# Result file: {result}
#   median S' coverage {sprime:.3f}, median S coverage {s:.3f}, S' - S >= 0.15 on {gap:.0%} of cases.
# {note}
# Only D_t / total_drug matters (dose is linear in the drug); the other keys are the goal 4.2 constants.
"""

NOTES = {
    "ok": "Strict band met.",
    "fallback_widened_band": "Strict band missed; the widened band (S' 0.65-0.85, S 0.25-0.55, gap on >= 70%) is met.",
    "fallback_gap_unmet": ("BOTH BANDS MISSED on the S' - S >= 0.15 criterion (the medians sit in the widened bands). The oracle "
                           "is only slightly better than script S on these tumours; raise at G1."),
    "no_feasible_point": "NO GRID POINT MEETS EVEN THE WIDENED MEDIANS. Best-effort point; raise at G1.",
}


def _pool_map(pool, workers):
    def run(fn, arg_tuples):
        if not arg_tuples:
            return []
        cols = list(zip(*arg_tuples))
        return list(pool.map(fn, *cols, chunksize=max(1, len(arg_tuples) // (workers * 8))))
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-dir", type=Path, default=ROOT / "data/v4_1/cases", help="L1 store root (<case>/case.npz)")
    ap.add_argument("--split", type=Path, default=ROOT / "config/v4_1/split.json")
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: every dev case in the store)")
    ap.add_argument("--config", type=Path, default=ROOT / "config/v4_1/world.yaml")
    ap.add_argument("--out", type=Path, default=None, help="world yaml to write (default: overwrite --config)")
    ap.add_argument("--result", type=Path, default=ROOT / "docs/research/v4.1-20261013/world_tuning.json")
    ap.add_argument("--drug-grid", default="", help="comma-separated total_drug values (default: 22 geometric 150..6e4)")
    ap.add_argument("--dt-grid", default="5", help="comma-separated D_t values")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--refine", type=int, default=14, help="stage 2 points between the best coarse neighbours (0 = off)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)

    split = load_split(args.split)
    if args.cases:
        ids = sorted(c for c in args.cases.split(",") if c)
    else:
        ids = sorted(p.parent.name for p in args.cases_dir.glob("*/case.npz") if p.parent.name in split["dev"])
    try:
        assert_dev_only(ids, split)
    except SplitError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    if not ids:
        print("REFUSED: no dev cases to tune on", file=sys.stderr)
        return 3
    seeds = [int(x) for x in args.seeds.split(",") if x]
    base = WorldConfig.from_yaml(args.config) if args.config.exists() else WorldConfig()
    drugs = [float(x) for x in args.drug_grid.split(",")] if args.drug_grid else [float(x) for x in np.geomspace(150, 60000, 22)]
    dts = [float(x) for x in args.dt_grid.split(",")]
    cases = {c: c for c in ids}  # workers load by id through the store adapter

    cache: dict = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(args.cases_dir), str(ROOT))) as pool:
        mp = _pool_map(pool, args.workers)
        res = grid_search(cases, base, drugs, dts, seeds=seeds, evaluate=episode_coverages_for_case, map_fn=mp, cache=cache)
        print(f"stage 1: {res['status']} best={res['best']}", flush=True)
        if args.refine > 0:
            order = sorted(set(drugs))
            k = order.index(res["best"]["total_drug"]) if res["best"]["total_drug"] in order else 0
            lo, hi = order[max(k - 1, 0)], order[min(k + 1, len(order) - 1)]
            fine = [float(x) for x in np.geomspace(lo, hi, args.refine)]
            drugs = sorted(set(drugs) | set(fine))
            res = grid_search(cases, base, drugs, dts, seeds=seeds, evaluate=episode_coverages_for_case, map_fn=mp, cache=cache)
            print(f"stage 2: {res['status']} best={res['best']}", flush=True)
        best = res["best"]
        cfg = base.with_(total_drug=best["total_drug"], dose_threshold=best["dose_threshold"])
        # timed re-run at the chosen point: determinism check against the grid values plus seconds per episode
        jobs = [(c, cfg, sd) for c in ids for sd in seeds]
        timed = list(pool.map(timed_episode_coverages_for_case, *zip(*jobs), chunksize=1))
    drift = max(abs(t[0] - res["per_case"][c]["s_by_seed"][seeds.index(sd)]) + abs(t[1] - res["per_case"][c]["sprime_by_seed"][seeds.index(sd)])
                for (c, _, sd), t in zip(jobs, timed))
    sec_s, sec_sp = [t[2] for t in timed], [t[3] for t in timed]
    per_case = [{"case": c, "split": "dev_a" if c in split["dev_a"] else "dev_b", **{k: (round(v, 6) if isinstance(v, float) else [round(x, 6) for x in v])
                 for k, v in res["per_case"][c].items()}, "gap": round(res["per_case"][c]["sprime"] - res["per_case"][c]["s"], 6)}
                for c in ids]
    gaps = np.array([p["gap"] for p in per_case])
    doc = {
        "status": res["status"], "band": res["band"], "targets": {"strict": STRICT, "widened": WIDENED},
        "seeds": seeds, "n_cases": len(ids), "dev_only": True,
        "chosen": {"total_drug": best["total_drug"], "dose_threshold": best["dose_threshold"],
                   "ratio_dt_over_drug": best["dose_threshold"] / best["total_drug"]},
        "summary": {"median_s": best["median_s"], "median_sprime": best["median_sprime"],
                    "gap_ge_0.15_fraction": best["gap_frac"], "median_gap": float(np.median(gaps)),
                    "min_gap": float(gaps.min()), "mean_s": float(np.mean([p["s"] for p in per_case])),
                    "mean_sprime": float(np.mean([p["sprime"] for p in per_case]))},
        "seconds_per_episode": {"script_S_mean": float(np.mean(sec_s)), "oracle_Sprime_mean": float(np.mean(sec_sp)),
                                "oracle_Sprime_max": float(np.max(sec_sp)), "n_episodes": len(timed)},
        "rerun_max_abs_drift": float(drift),
        "per_case": per_case,
        "grid": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items()} for p in res["points"]],
        "notes": "Per-case coverage is the mean over seeds; per-seed values are in s_by_seed / sprime_by_seed. Aggregates and "
                 "public MSD case ids only.",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(doc, indent=1) + "\n")
    out = args.out or args.config
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        rel = args.result.resolve().relative_to(ROOT)
    except ValueError:
        rel = args.result
    out.write_text(HEADER.format(seeds=",".join(map(str, seeds)), status=res["status"], band=res["band"], result=rel,
                                 sprime=best["median_sprime"], s=best["median_s"], gap=best["gap_frac"],
                                 note=NOTES[res["status"]]) + yaml.safe_dump({"world": cfg.to_dict()}, sort_keys=False))
    print(f"{res['status']}: {doc['chosen']} {doc['summary']} sec/ep={doc['seconds_per_episode']} drift={drift:.2e}")
    return 0 if res["status"] == "ok" else (1 if res["status"] == "fallback_widened_band" else 2)


if __name__ == "__main__":
    sys.exit(main())
